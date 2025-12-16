namespace LocalLLMChat.Models;

public class InferenceDiagnostics
{
    // Timing
    public DateTime InferenceStartTime { get; set; }
    public DateTime? FirstTokenTime { get; set; }
    public DateTime? LastTokenTime { get; set; }
    public double TimeToFirstTokenMs { get; set; }
    public double TotalInferenceTimeMs { get; set; }

    // Token Stats
    public int PromptTokenCount { get; set; }
    public int GeneratedTokenCount { get; set; }
    public double TokensPerSecond { get; set; }
    public double PromptProcessingSpeed { get; set; } // tokens/sec for prompt eval

    // Context
    public int ContextSize { get; set; }
    public int ContextUsed { get; set; }
    public double ContextUtilization { get; set; } // percentage

    // Sampling Info
    public float Temperature { get; set; }
    public float TopP { get; set; }
    public int TopK { get; set; }
    public float MinP { get; set; }
    public float RepeatPenalty { get; set; }

    // Memory (estimated)
    public long ModelMemoryBytes { get; set; }
    public long ContextMemoryBytes { get; set; }
    public string ModelMemoryFormatted => FormatBytes(ModelMemoryBytes);
    public string ContextMemoryFormatted => FormatBytes(ContextMemoryBytes);

    // Token Stream (last N tokens with their info)
    public List<TokenInfo> RecentTokens { get; set; } = new();

    // Entropy/Perplexity estimates
    public double AverageTokenEntropy { get; set; }
    public double EstimatedPerplexity { get; set; }

    // Status
    public bool IsGenerating { get; set; }
    public string Status { get; set; } = "Idle";

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2} GB";
    }
}

public class TokenInfo
{
    public int TokenId { get; set; }
    public string TokenText { get; set; } = string.Empty;
    public float Probability { get; set; }
    public float LogProbability { get; set; }
    public double Entropy { get; set; }
    public DateTime Timestamp { get; set; }
    public double GenerationTimeMs { get; set; }

    // Top alternatives considered
    public List<TokenCandidate> TopCandidates { get; set; } = new();
}

public class TokenCandidate
{
    public int TokenId { get; set; }
    public string TokenText { get; set; } = string.Empty;
    public float Probability { get; set; }
    public float LogProbability { get; set; }
}

public class LayerActivation
{
    public int LayerIndex { get; set; }
    public string LayerName { get; set; } = string.Empty;
    public float MeanActivation { get; set; }
    public float MaxActivation { get; set; }
    public float MinActivation { get; set; }
    public float Variance { get; set; }
    public float L2Norm { get; set; }
}
