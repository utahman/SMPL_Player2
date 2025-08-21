// SMPL_Player.cs
// Player that applies SMPL motion frames (276-dim format) to a Unity Humanoid rigged figure.
// Handles coordinate conversion (SMPL Z-up RHS -> Unity Y-up LHS)
// and retargeting (SMPL A-Pose -> Unity T-Pose).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor; // Used only to read DefaultAsset bytes in Editor
#endif

public class SMPL_Player : MonoBehaviour
{
    [Header("Inputs")]
    [Tooltip("Humanoid animator. MUST be in T-Pose when the scene starts for correct calibration.")]
    public Animator targetAnimator;
    [Tooltip("JSON with SMPL rest pose (A-Pose: stand_pose_flat) for calibration.")]
    public TextAsset standPoseJson;
    [Tooltip("Drag a .npy / .bytes / .data asset here. Expected shape [1, F, 276] or [F, 276].")]
    public UnityEngine.Object framesFile;

    // ==== Data layout ====
    private const int FEATURE_DIM = 276;
    private const int NUM_JOINTS = 22;

    // Expected layout per frame:
    private const int TRANSL_OFFSET = 0;                                     // [0..2] Root translation
    private const int POSES_6D_OFFSET = TRANSL_OFFSET + 3;                   // [3..134] 6D local rotations

    // Mapping SMPL joint index -> Unity Humanoid bone (Dynamically initialized)
    private readonly Dictionary<int, HumanBodyBones> m_JointMap = new Dictionary<int, HumanBodyBones>();

    // Playback queue
    private readonly Queue<HumanoidPose> m_MotionQueue = new Queue<HumanoidPose>();

    // Retarget calibration (A-Pose to T-Pose offsets)
    // Stores the offset rotation for each joint: q_offset = qT * Inverse(qA)
    private readonly Quaternion[] m_RetargetOffsets = new Quaternion[NUM_JOINTS];
    private readonly bool[] m_BoneAvailable = new bool[NUM_JOINTS];
    private bool m_CalibrationReady = false;

    private Quaternion m_InitialRootRotation;
    private Vector3 m_InitialRootPosition;

    // ==== Coordinate system helpers ====
    // Input Assumption (based on original script context): SMPL is Z-up, Right-Handed (RH).
    // Target: Unity is Y-up, Left-Handed (LH).
    // The transformation is a swap of Y and Z axes (Permutation (x, z, y)). det(C)=-1.

    // Helper to convert SMPL position (Z-up, RH) to Unity position (Y-up, LH)
    // Unity_X = SMPL_X, Unity_Y = SMPL_Z, Unity_Z = SMPL_Y
    private static Vector3 ConvertSmplToUnityPosition(Vector3 vSmpl)
    {
        return new Vector3(vSmpl.x, vSmpl.z, vSmpl.y);
    }

    // Helper to convert SMPL rotation (Z-up, RH) to Unity rotation (Y-up, LH)
    // This handles both the axis swap and the handedness change inherent in the permutation.
    // Q_U = (-x, -z, -y, w).
    private static Quaternion ConvertSmplToUnityRotation(Quaternion qSmpl)
    {
        return new Quaternion(-qSmpl.x, -qSmpl.z, -qSmpl.y, qSmpl.w);
    }

    // ==== Types ====
    [Serializable]
    private class StandPoseData
    {
        public int history_length;
        public int feature_dim;
        public float[] stand_pose_flat; // The A-Pose data
    }

    private struct HumanoidPose
    {
        public Vector3 RootPositionUnity;
        public Dictionary<HumanBodyBones, Quaternion> BoneLocals;
    }

    // ======================== Unity lifecycle ========================
    private void Start()
    {
        Application.targetFrameRate = 30; // Assuming 30 FPS data

        if (!CheckInputs())
        {
            enabled = false;
            return;
        }

        m_InitialRootRotation = targetAnimator.transform.rotation;
        m_InitialRootPosition = targetAnimator.transform.position;

        // IMPORTANT: We assume the Animator is currently in the T-Pose.
        Debug.Log("[SMPL_Player] Assuming Target Animator is in T-Pose for calibration.");
        // Ensure the animator is initialized to its bind pose.
        targetAnimator.Update(0f);

        InitializeBoneMappingAndAvailability();

        // 1) Calibrate retargeting using the stand pose (A-Pose) against the current T-Pose.
        if (CalibrateFromStandPose())
        {
            m_CalibrationReady = true;
            // 2) Load motion and queue playback
            LoadAndQueueMotion();
        }
        else
        {
            Debug.LogError("[SMPL_Player] Calibration failed. Cannot play motion.");
            enabled = false;
        }
    }

    // Use LateUpdate to ensure we override any Animator updates for the frame.
    private void LateUpdate()
    {
        if (!m_CalibrationReady || m_MotionQueue.Count == 0)
            return;

        var pose = m_MotionQueue.Dequeue();
        ApplyPoseToHumanoid(pose);
    }

    // ======================== Initialization & Calibration ========================

    private bool CheckInputs()
    {
        if (!targetAnimator || !targetAnimator.isHuman)
        {
            Debug.LogError("[SMPL_Player] Target Animator is not set or not Humanoid.");
            return false;
        }
        if (!standPoseJson)
        {
            Debug.LogError("[SMPL_Player] Stand Pose JSON (A-Pose) is not assigned. Calibration requires this.");
            return false;
        }
        if (!framesFile)
        {
            Debug.LogError("[SMPL_Player] Frames file is not assigned.");
            return false;
        }

        // Ensure the Animator isn't applying its own root motion.
        if (targetAnimator.applyRootMotion)
        {
            Debug.LogWarning("[SMPL_Player] Disabling 'Apply Root Motion' on the Animator as it is controlled by this script.");
            targetAnimator.applyRootMotion = false;
        }

        return true;
    }

    // Adaptively map SMPL joints to Unity Humanoid bones based on availability.
    private void InitializeBoneMappingAndAvailability()
    {
        m_JointMap.Clear();
        Array.Clear(m_BoneAvailable, 0, m_BoneAvailable.Length);

        // Standard mapping (excluding spine)
        var map = new Dictionary<int, HumanBodyBones>
        {
            {0, HumanBodyBones.Hips},
            {1, HumanBodyBones.LeftUpperLeg}, {2, HumanBodyBones.RightUpperLeg},
            {4, HumanBodyBones.LeftLowerLeg}, {5, HumanBodyBones.RightLowerLeg},
            {7, HumanBodyBones.LeftFoot}, {8, HumanBodyBones.RightFoot},
            {10, HumanBodyBones.LeftToes}, {11, HumanBodyBones.RightToes},
            {12, HumanBodyBones.Neck},
            {13, HumanBodyBones.LeftShoulder}, {14, HumanBodyBones.RightShoulder},
            {15, HumanBodyBones.Head},
            {16, HumanBodyBones.LeftUpperArm}, {17, HumanBodyBones.RightUpperArm},
            {18, HumanBodyBones.LeftLowerArm}, {19, HumanBodyBones.RightLowerArm},
            {20, HumanBodyBones.LeftHand}, {21, HumanBodyBones.RightHand}
        };

        // Adaptive Spine mapping (SMPL joints: 3=spine1, 6=spine2, 9=spine3)
        bool hasSpine = targetAnimator.GetBoneTransform(HumanBodyBones.Spine) != null;
        bool hasChest = targetAnimator.GetBoneTransform(HumanBodyBones.Chest) != null;
        bool hasUpperChest = targetAnimator.GetBoneTransform(HumanBodyBones.UpperChest) != null;

        if (hasSpine)
        {
            map[3] = HumanBodyBones.Spine;
        }

        if (hasChest && hasUpperChest)
        {
            // Ideal case: Map all three SMPL spine joints.
            map[6] = HumanBodyBones.Chest;
            map[9] = HumanBodyBones.UpperChest;
        }
        else if (hasChest)
        {
            // Common case: Spine and Chest only. Map SMPL spine1 (3) and spine3 (9), ignore spine2 (6).
            map[9] = HumanBodyBones.Chest;
            Debug.Log("[SMPL_Player] UpperChest missing. Mapping SMPL spine3 (9) to Chest, ignoring spine2 (6).");
        }
        else if (hasSpine)
        {
            Debug.LogWarning("[SMPL_Player] Chest bone missing. Spine fidelity will be reduced.");
        }

        // Finalize map and check availability
        for (int j = 0; j < NUM_JOINTS; j++)
        {
            m_RetargetOffsets[j] = Quaternion.identity;
            if (map.TryGetValue(j, out var bone))
            {
                if (targetAnimator.GetBoneTransform(bone) != null)
                {
                    m_JointMap[j] = bone;
                    m_BoneAvailable[j] = true;
                }
            }
        }
    }


    private bool CalibrateFromStandPose()
    {
        if (!TryGetStandPose(out var sp))
        {
            // Error already logged in TryGetStandPose
            return false;
        }

        if (sp.feature_dim != FEATURE_DIM)
            Debug.LogWarning($"[SMPL_Player] Stand pose feature_dim is {sp.feature_dim}, expected {FEATURE_DIM}.");

        // Stand pose data shape normalization
        int history_length = sp.history_length > 0 ? sp.history_length : 1;
        if (sp.stand_pose_flat.Length < history_length * sp.feature_dim)
        {
            Debug.LogError("[SMPL_Player] Stand pose data length mismatch.");
            return false;
        }

        var shape = new[] { 1, history_length, sp.feature_dim };

        // Use the last frame of the stand pose sequence
        int f = Mathf.Clamp(history_length - 1, 0, history_length - 1);

        // Calibrate mapping from stand pose (A-Pose) to current animator pose (T-Pose)
        CalibrateOffsets(sp.stand_pose_flat, shape, f);
        return true;
    }

    // Calculate m_RetargetOffsets so that SMPL A-Pose maps into Unity Humanoid T-Pose
    // Uses the Simple Offset method: q_offset = qT * Inverse(qA)
    private void CalibrateOffsets(float[] smplRestPoseData, int[] sh, int f)
    {
        // 1. Get SMPL Rest Pose (A-Pose) local rotations and convert to Unity coordinates
        Quaternion[] smplAPoseUnity = new Quaternion[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; j++)
        {
            if (!m_BoneAvailable[j])
            {
                smplAPoseUnity[j] = Quaternion.identity;
                continue;
            }

            float[] d6 = Read6D(smplRestPoseData, sh, f, POSES_6D_OFFSET + j * 6);
            Quaternion qSmpl = SixDToQuat(d6);
            // Crucially, convert this A-Pose rotation to the Unity coordinate system
            smplAPoseUnity[j] = ConvertSmplToUnityRotation(qSmpl);
        }

        // 2. Get Unity Bind Pose (T-Pose) local rotations
        // We rely on the Animator being currently in the T-Pose.
        Quaternion[] unityTPose = new Quaternion[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; j++)
        {
            if (!m_JointMap.TryGetValue(j, out var bone) || !m_BoneAvailable[j])
            {
                unityTPose[j] = Quaternion.identity;
                continue;
            }
            var t = targetAnimator.GetBoneTransform(bone);
            // Read the local rotation of the bone in the T-Pose.
            unityTPose[j] = t.localRotation;
        }

        // 3. Calculate offsets: q_offset = qT * Inverse(qA)
        for (int j = 0; j < NUM_JOINTS; j++)
        {
            if (!m_BoneAvailable[j])
                continue;

            // This calculation ensures that when the animation is at the A-Pose (q_anim = qA),
            // the final rotation applied is the T-Pose (qT).
            m_RetargetOffsets[j] = unityTPose[j] * Quaternion.Inverse(smplAPoseUnity[j]);
        }

        Debug.Log("[SMPL_Player] Retarget offsets computed (A-Pose to T-Pose).");
    }

    // ======================== Motion Loading and Playback ========================

    private void LoadAndQueueMotion()
    {
        if (!TryLoadFramesFromObject(framesFile, out var flat, out var shape, FEATURE_DIM))
        {
            Debug.LogError("[SMPL_Player] Could not read frames.");
            return;
        }

        // Validate and normalize shape: Allow [1, F, D] or [F, D]
        if ((shape.Length == 3 && shape[0] == 1 && shape[2] == FEATURE_DIM) ||
            (shape.Length == 2 && shape[1] == FEATURE_DIM))
        {
            // Reshape to [1, F, D] for consistency (Idx function expects 3 dims)
            if (shape.Length == 2)
            {
                shape = new[] { 1, shape[0], shape[1] };
            }
        }
        else
        {
            Debug.LogError($"[SMPL_Player] Invalid frames shape: [{string.Join(", ", shape)}]. Expected [1, F, {FEATURE_DIM}] or [F, {FEATURE_DIM}].");
            return;
        }


        int frames = shape[1];
        for (int f = 0; f < frames; f++)
            m_MotionQueue.Enqueue(BuildPoseFromFrame(flat, shape, f));

        Debug.Log($"[SMPL_Player] Queued {frames} frames for playback.");

        // Apply the first frame immediately to set the initial pose
        if (m_MotionQueue.Count > 0)
        {
            ApplyPoseToHumanoid(m_MotionQueue.Peek());
        }
    }

    // The core processing pipeline for a single frame
    private HumanoidPose BuildPoseFromFrame(float[] src, int[] sh, int f)
    {
        var pose = new HumanoidPose
        {
            RootPositionUnity = Vector3.zero,
            BoneLocals = new Dictionary<HumanBodyBones, Quaternion>()
        };

        // --- Root Translation ---
        Vector3 smplRoot = ReadVec3(src, sh, f, TRANSL_OFFSET);
        // Convert coordinates (Z-up RH -> Y-up LH)
        pose.RootPositionUnity = ConvertSmplToUnityPosition(smplRoot);

        // --- Joints Rotations ---
        for (int j = 0; j < NUM_JOINTS; j++)
        {
            if (!m_JointMap.TryGetValue(j, out var bone) || !m_BoneAvailable[j])
                continue;

            float[] d6 = Read6D(src, sh, f, POSES_6D_OFFSET + j * 6);

            // 1. 6D to Quaternion (in SMPL space)
            Quaternion qSmpl = SixDToQuat(d6);

            // 2. Convert coordinate system (Z-up RH -> Y-up LH)
            Quaternion qUnityAnim = ConvertSmplToUnityRotation(qSmpl);

            // 3. Retarget (A-Pose -> T-Pose)
            // q_final_local = q_offset * q_anim
            Quaternion finalLocalRotation = m_RetargetOffsets[j] * qUnityAnim;

            pose.BoneLocals[bone] = finalLocalRotation;
        }

        return pose;
    }

    private void ApplyPoseToHumanoid(HumanoidPose p)
    {
        if (!targetAnimator)
            return;

        // Apply Root Motion
        // Position is driven by the converted SMPL translation, relative to the starting position.
        targetAnimator.transform.position = m_InitialRootPosition + p.RootPositionUnity;

        // Rotation of the root GameObject is kept fixed relative to the start orientation.
        // The character's body orientation is driven by the Hips local rotation.
        targetAnimator.transform.rotation = m_InitialRootRotation;

        // Apply Bone Rotations (Local space)
        foreach (var kv in p.BoneLocals)
        {
            var t = targetAnimator.GetBoneTransform(kv.Key);
            if (t)
                t.localRotation = kv.Value;
        }
    }

    // ======================== Math & array helpers ========================

    private static Quaternion SixDToQuat(float[] d6) => SixDToMatrix(d6).rotation;

    // FIX: Standard 6D representation (Zhou et al.) defines the first two COLUMNS, not rows.
    private static Matrix4x4 SixDToMatrix(float[] d6)
    {
        // 6D representation = first two columns of rotation matrix; rebuild with Gram–Schmidt process.

        // Extract the two column vectors
        Vector3 c0a = new Vector3(d6[0], d6[1], d6[2]);
        Vector3 c1a = new Vector3(d6[3], d6[4], d6[5]);

        // Gram–Schmidt orthogonalization
        Vector3 c0 = c0a.normalized;
        // c1 = normalize(c1a - dot(c0, c1a) * c0)
        Vector3 c1 = (c1a - Vector3.Dot(c0, c1a) * c0).normalized;

        // Calculate the third column using cross product. Assumes RHS source data.
        Vector3 c2 = Vector3.Cross(c0, c1);

        // Construct the rotation matrix by setting the columns (Unity Matrix4x4 is column-major)
        Matrix4x4 m = Matrix4x4.identity;
        m.SetColumn(0, new Vector4(c0.x, c0.y, c0.z, 0));
        m.SetColumn(1, new Vector4(c1.x, c1.y, c1.z, 0));
        m.SetColumn(2, new Vector4(c2.x, c2.y, c2.z, 0));
        return m;
    }

    // Helper for accessing flattened 3D array [B, F, D] (C-order)
    private static int Idx(int[] sh, int b, int f, int d)
    {
        // Assumes sh has 3 dimensions: [Batch, Frames, Dim]
        return (b * sh[1] + f) * sh[2] + d;
    }

    private static Vector3 ReadVec3(float[] src, int[] sh, int f, int off)
    {
        return new Vector3(
            src[Idx(sh, 0, f, off + 0)],
            src[Idx(sh, 0, f, off + 1)],
            src[Idx(sh, 0, f, off + 2)]
        );
    }

    private static float[] Read6D(float[] src, int[] sh, int f, int off)
    {
        var a = new float[6];
        for (int k = 0; k < 6; k++)
            a[k] = src[Idx(sh, 0, f, off + k)];
        return a;
    }

    // ======================== File / JSON helpers ========================

    private bool TryGetStandPose(out StandPoseData sp)
    {
        sp = null;
        if (!standPoseJson)
            return false;

        try
        {
            string jsonText = standPoseJson.text;

            // Unity's JsonUtility might fail on large arrays, so we parse the main object first
            sp = JsonUtility.FromJson<StandPoseData>(jsonText);
            if (sp == null)
                return false;

            // Manually parse the large 'stand_pose_flat' array using Regex if JsonUtility didn't fill it
            if (sp.stand_pose_flat == null || sp.stand_pose_flat.Length == 0)
            {
                var match = Regex.Match(jsonText, @"\""stand_pose_flat\"":\s*\[([^\]]+)\]");
                if (!match.Success)
                {
                    Debug.LogError("[SMPL_Player] Could not find 'stand_pose_flat' array in JSON.");
                    return false;
                }

                string[] floatStrings = match.Groups[1].Value.Split(',');
                List<float> floats = new List<float>(floatStrings.Length);
                foreach (string floatStr in floatStrings)
                {
                    // Use InvariantCulture for robust float parsing (handles '.')
                    if (float.TryParse(floatStr.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float val))
                    {
                        floats.Add(val);
                    }
                }
                sp.stand_pose_flat = floats.ToArray();
            }

            return sp.stand_pose_flat.Length > 0;
        }
        catch (Exception e)
        {
            Debug.LogError($"[SMPL_Player] Stand-pose JSON parse error: {e.Message}");
            return false;
        }
    }

    private bool TryLoadFramesFromObject(UnityEngine.Object obj, out float[] flat, out int[] shape, int expectedLastDim)
    {
        flat = null;
        shape = null;

        if (obj == null)
            return false;

        if (!TryGetBytesFromObject(obj, out var bytes, out var pathHint))
            return false;

        // Try NPY first
        if (TryParseNpy(bytes, out flat, out shape, expectedLastDim))
        {
            Debug.Log($"[SMPL_Player] Loaded NPY frames ({(string.IsNullOrEmpty(pathHint) ? "<memory>" : pathHint)}): shape [{string.Join(", ", shape)}].");
            return true;
        }

        // RAW fallback: expect contiguous float32 (C-order)
        if (TryParseRawFloat32(bytes, expectedLastDim, out flat, out shape))
        {
            Debug.Log($"[SMPL_Player] Loaded RAW float32 frames ({(string.IsNullOrEmpty(pathHint) ? "<memory>" : pathHint)}): shape [{string.Join(", ", shape)}].");
            return true;
        }

        Debug.LogError("[SMPL_Player] Failed to parse frames as NPY or RAW float32.");
        return false;
    }

    private bool TryGetBytesFromObject(UnityEngine.Object obj, out byte[] bytes, out string pathHint)
    {
        bytes = null;
        pathHint = null;

        // TextAsset path (works in Editor & Player)
        if (obj is TextAsset ta)
        {
            bytes = ta.bytes;
            pathHint = ta.name;
            return bytes != null && bytes.Length > 0;
        }

        // Editor-only: allow dragging arbitrary files (DefaultAsset) like sample.data
#if UNITY_EDITOR
        string p = AssetDatabase.GetAssetPath(obj);
        if (!string.IsNullOrEmpty(p) && File.Exists(p))
        {
            try
            {
                bytes = File.ReadAllBytes(p);
                pathHint = p;
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[SMPL_Player] Could not read asset bytes at '{p}': {e.Message}");
                return false;
            }
        }
#endif

        Debug.LogError("[SMPL_Player] Unsupported frames object. Use a TextAsset (.bytes / .npy.bytes) or drag a file asset in Editor.");
        return false;
    }

    // Minimal NPY parser for float32, C-order arrays.
    private static bool TryParseNpy(byte[] npyBytes, out float[] flat, out int[] shape, int expectedLastDim)
    {
        flat = null;
        shape = null;

        try
        {
            using (var ms = new MemoryStream(npyBytes))
            using (var br = new BinaryReader(ms))
            {
                // Magic
                if (br.ReadByte() != 0x93 || System.Text.Encoding.ASCII.GetString(br.ReadBytes(5)) != "NUMPY")
                    return false;

                byte major = br.ReadByte();
                br.ReadByte(); // minor

                int headerLen = (major == 1) ? br.ReadUInt16() : (major >= 2 ? br.ReadInt32() : 0);
                if (headerLen == 0)
                    return false;

                string header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(headerLen));

                // shape
                var mShape = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)");
                if (!mShape.Success)
                    return false;

                var parts = mShape.Groups[1].Value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                                                    .Select(s => s.Trim())
                                                    .Where(s => !string.IsNullOrEmpty(s))
                                                    .Select(int.Parse)
                                                    .ToArray();
                shape = parts;

                // Validation of specific shape happens in LoadAndQueueMotion (we allow [1,F,D] or [F,D] here)

                // fortran_order must be False (C-order)
                var mFortran = Regex.Match(header, @"'fortran_order'\s*:\s*(True|False)");
                if (mFortran.Success && mFortran.Groups[1].Value == "True")
                {
                    Debug.LogError("[SMPL_Player] NPY is Fortran-order. Convert to C-order.");
                    return false;
                }

                // descr check (float32)
                if (!Regex.IsMatch(header, @"'descr'\s*:\s*'([<|\|])\s*f4'"))
                {
                    Debug.LogWarning($"[SMPL_Player] NPY 'descr' is not recognized as f32. Attempting to read anyway.");
                }

                long count = 1;
                foreach (var dim in shape)
                    count *= dim;

                int byteCount = checked((int)(count * 4));
                var dataBytes = br.ReadBytes(byteCount);
                if (dataBytes.Length != byteCount)
                {
                    Debug.LogError("[SMPL_Player] NPY data truncated.");
                    return false;
                }

                flat = new float[count];
                Buffer.BlockCopy(dataBytes, 0, flat, 0, dataBytes.Length);
                return true;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SMPL_Player] NPY parse failed: {e.Message}");
            return false;
        }
    }

    private static bool TryParseRawFloat32(byte[] rawBytes, int expectedLastDim, out float[] flat, out int[] shape)
    {
        flat = null;
        shape = null;

        try
        {
            if (rawBytes.Length % 4 != 0)
                return false;

            int totalFloats = rawBytes.Length / 4;
            if (totalFloats % expectedLastDim != 0)
                return false;

            int frames = totalFloats / expectedLastDim;
            // We return shape as [F, D] for raw files, handled in LoadAndQueueMotion
            shape = new[] { frames, expectedLastDim };

            flat = new float[totalFloats];
            Buffer.BlockCopy(rawBytes, 0, flat, 0, rawBytes.Length);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }
}
