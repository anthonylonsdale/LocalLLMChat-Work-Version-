using System.Text.Json.Serialization;

namespace LocalLLMChat.Models;

public class LlmSettings
{
    // Model Loading / Runtime configuration
    public string ModelPath { get; set; } = string.Empty;
    public bool CpuOnly { get; set; } = true;
    public int ContextLength { get; set; } = 4096;
    public int GpuOffloadLayers { get; set; } = 0;
    public int CpuThreadPoolSize { get; set; } = -1; // -1/0 = auto
    public int EvaluationBatchSize { get; set; } = 512;
    public bool RopeFrequencyBaseEnabled { get; set; }
    public float? RopeFrequencyBase { get; set; }
    public bool RopeFrequencyScaleEnabled { get; set; }
    public float? RopeFrequencyScale { get; set; }
    public bool OffloadKvCacheToGpu { get; set; } = true;
    public bool KeepModelInMemory { get; set; } = true;
    public bool TryMmap { get; set; } = true;
    public bool UseMlock { get; set; } = false;
    public bool RandomSeed { get; set; } = true;
    public int Seed { get; set; } = 0;
    public bool FlashAttention { get; set; } = true;
    public bool KCacheQuantTypeEnabled { get; set; }
    public string? KCacheQuantType { get; set; }
    public bool VCacheQuantTypeEnabled { get; set; }
    public string? VCacheQuantType { get; set; }
    public string OverrideDomainType { get; set; } = "No Override";

    // Generation Settings
    public int MaxTokens { get; set; } = 1024;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 40;
    public float MinP { get; set; } = 0.05f;
    public float RepeatPenalty { get; set; } = 1.1f;
    public int RepeatLastN { get; set; } = 64; // Tokens to consider for repeat penalty
    public float FrequencyPenalty { get; set; } = 0.0f;
    public float PresencePenalty { get; set; } = 0.0f;
    public int RuntimeSeed { get; set; } = -1; // -1 = random (used for sampling)

    // System Prompt
    public string SystemPrompt { get; set; } =
        "You are a helpful AI assistant. You provide clear, accurate, and concise responses to user questions.";

    // Stop Sequences
    public List<string> StopSequences { get; set; } = new() { "User:", "\nUser:", "Human:", "\nHuman:" };

    // Compatibility helpers for existing code paths / bindings
    [JsonIgnore]
    public uint ContextSize
    {
        get => (uint)Math.Max(ContextLength, 1);
        set => ContextLength = (int)value;
    }

    [JsonIgnore]
    public int GpuLayerCount
    {
        get => GpuOffloadLayers;
        set => GpuOffloadLayers = value;
    }

    [JsonIgnore]
    public uint BatchSize
    {
        get => (uint)Math.Max(EvaluationBatchSize, 1);
        set => EvaluationBatchSize = (int)value;
    }

    [JsonIgnore]
    public int ThreadCount
    {
        get => CpuThreadPoolSize;
        set => CpuThreadPoolSize = value;
    }

    [JsonIgnore]
    public bool UseFlashAttention
    {
        get => FlashAttention;
        set => FlashAttention = value;
    }

    public LlmSettings Clone()
    {
        var clone = (LlmSettings)MemberwiseClone();
        clone.StopSequences = new List<string>(StopSequences);
        return clone;
    }
}

// Runtime settings that can be changed without reloading the model
public class RuntimeLlmSettings
{
    public int MaxTokens { get; set; }
    public float Temperature { get; set; }
    public float TopP { get; set; }
    public int TopK { get; set; }
    public float MinP { get; set; }
    public float RepeatPenalty { get; set; }
    public int RepeatLastN { get; set; }
    public float FrequencyPenalty { get; set; }
    public float PresencePenalty { get; set; }
    public int Seed { get; set; }
    public string SystemPrompt { get; set; } = string.Empty;
    public List<string> StopSequences { get; set; } = new();
}
