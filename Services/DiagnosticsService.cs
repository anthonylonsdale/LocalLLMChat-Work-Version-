using LocalLLMChat.Models;
using System.Collections.Concurrent;

namespace LocalLLMChat.Services;

public interface IDiagnosticsService
{
    InferenceDiagnostics CurrentDiagnostics { get; }
    event EventHandler<InferenceDiagnostics>? DiagnosticsUpdated;

    void StartInference(int promptTokens, RuntimeLlmSettings settings, uint contextSize);
    void RecordToken(int tokenId, string tokenText, float probability, List<TokenCandidate>? topCandidates = null);
    void EndInference();
    void UpdateMemoryStats(long modelMemory, long contextMemory);
    void Reset();
}

public class DiagnosticsService : IDiagnosticsService
{
    private readonly ILogger<DiagnosticsService> _logger;
    private InferenceDiagnostics _current = new();
    private DateTime _lastTokenTime;
    private readonly object _lock = new();
    private const int MaxRecentTokens = 50;

    public InferenceDiagnostics CurrentDiagnostics
    {
        get
        {
            lock (_lock)
            {
                return _current;
            }
        }
    }

    public event EventHandler<InferenceDiagnostics>? DiagnosticsUpdated;

    public DiagnosticsService(ILogger<DiagnosticsService> logger)
    {
        _logger = logger;
    }

    public void StartInference(int promptTokens, RuntimeLlmSettings settings, uint contextSize)
    {
        lock (_lock)
        {
            _current = new InferenceDiagnostics
            {
                InferenceStartTime = DateTime.UtcNow,
                PromptTokenCount = promptTokens,
                ContextSize = (int)contextSize,
                Temperature = settings.Temperature,
                TopP = settings.TopP,
                TopK = settings.TopK,
                MinP = settings.MinP,
                RepeatPenalty = settings.RepeatPenalty,
                IsGenerating = true,
                Status = "Processing prompt..."
            };
            _lastTokenTime = _current.InferenceStartTime;
        }

        _logger.LogDebug("Inference started: {PromptTokens} prompt tokens, context size {ContextSize}",
            promptTokens, contextSize);

        NotifyUpdate();
    }

    public void RecordToken(int tokenId, string tokenText, float probability, List<TokenCandidate>? topCandidates = null)
    {
        lock (_lock)
        {
            var now = DateTime.UtcNow;
            var tokenGenTime = (now - _lastTokenTime).TotalMilliseconds;

            // First token
            if (_current.FirstTokenTime == null)
            {
                _current.FirstTokenTime = now;
                _current.TimeToFirstTokenMs = (now - _current.InferenceStartTime).TotalMilliseconds;
                _current.PromptProcessingSpeed = _current.PromptTokenCount / (_current.TimeToFirstTokenMs / 1000.0);
                _current.Status = "Generating...";
            }

            _current.LastTokenTime = now;
            _current.GeneratedTokenCount++;

            // Calculate tokens per second
            var totalGenTime = (now - (_current.FirstTokenTime ?? now)).TotalSeconds;
            if (totalGenTime > 0)
            {
                _current.TokensPerSecond = _current.GeneratedTokenCount / totalGenTime;
            }

            // Calculate entropy from probability
            var logProb = probability > 0 ? MathF.Log(probability) : -100f;
            var entropy = -probability * logProb;

            // Update context utilization
            _current.ContextUsed = _current.PromptTokenCount + _current.GeneratedTokenCount;
            _current.ContextUtilization = (_current.ContextUsed / (double)_current.ContextSize) * 100;

            // Create token info
            var tokenInfo = new TokenInfo
            {
                TokenId = tokenId,
                TokenText = tokenText,
                Probability = probability,
                LogProbability = logProb,
                Entropy = entropy,
                Timestamp = now,
                GenerationTimeMs = tokenGenTime,
                TopCandidates = topCandidates ?? new List<TokenCandidate>()
            };

            _current.RecentTokens.Add(tokenInfo);

            // Keep only recent tokens
            if (_current.RecentTokens.Count > MaxRecentTokens)
            {
                _current.RecentTokens.RemoveAt(0);
            }

            // Update average entropy
            if (_current.RecentTokens.Count > 0)
            {
                _current.AverageTokenEntropy = _current.RecentTokens.Average(t => t.Entropy);
                // Perplexity = exp(average negative log probability)
                var avgNegLogProb = _current.RecentTokens.Average(t => -t.LogProbability);
                _current.EstimatedPerplexity = Math.Exp(avgNegLogProb);
            }

            _current.TotalInferenceTimeMs = (now - _current.InferenceStartTime).TotalMilliseconds;
            _lastTokenTime = now;
        }

        NotifyUpdate();
    }

    public void EndInference()
    {
        lock (_lock)
        {
            _current.IsGenerating = false;
            _current.Status = "Complete";
            _current.TotalInferenceTimeMs = (DateTime.UtcNow - _current.InferenceStartTime).TotalMilliseconds;
        }

        _logger.LogDebug("Inference complete: {GeneratedTokens} tokens in {TotalTime:F0}ms ({TokensPerSec:F1} t/s)",
            _current.GeneratedTokenCount, _current.TotalInferenceTimeMs, _current.TokensPerSecond);

        NotifyUpdate();
    }

    public void UpdateMemoryStats(long modelMemory, long contextMemory)
    {
        lock (_lock)
        {
            _current.ModelMemoryBytes = modelMemory;
            _current.ContextMemoryBytes = contextMemory;
        }

        NotifyUpdate();
    }

    public void Reset()
    {
        lock (_lock)
        {
            _current = new InferenceDiagnostics();
        }

        NotifyUpdate();
    }

    private void NotifyUpdate()
    {
        try
        {
            DiagnosticsUpdated?.Invoke(this, CurrentDiagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error notifying diagnostics update");
        }
    }
}
