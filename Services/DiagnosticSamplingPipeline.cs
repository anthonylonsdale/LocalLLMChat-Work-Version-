using LLama.Native;
using LLama.Sampling;
using LocalLLMChat.Models;

namespace LocalLLMChat.Services;

/// <summary>
/// A sampling pipeline wrapper that captures token probabilities for diagnostics.
/// Wraps the DefaultSamplingPipeline and intercepts sampling to extract real probability data.
/// Compatible with LLamaSharp 0.25.0 ISamplingPipeline interface.
/// </summary>
public class DiagnosticSamplingPipeline : ISamplingPipeline
{
    private readonly DefaultSamplingPipeline _innerPipeline;
    private readonly IDiagnosticsService _diagnostics;
    private readonly Action<TokenProbabilityInfo>? _onTokenSampled;

    // Store the last logits for probability calculation after sampling
    private float[]? _lastLogits;
    private SafeLLamaContextHandle? _lastContext;

    // Last sampled token info
    public TokenProbabilityInfo? LastTokenInfo { get; private set; }

    public DiagnosticSamplingPipeline(
        DefaultSamplingPipeline innerPipeline,
        IDiagnosticsService diagnostics,
        Action<TokenProbabilityInfo>? onTokenSampled = null)
    {
        _innerPipeline = innerPipeline;
        _diagnostics = diagnostics;
        _onTokenSampled = onTokenSampled;
    }

    public void Accept(LLamaToken token)
    {
        _innerPipeline.Accept(token);
    }

    public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray data)
    {
        _innerPipeline.Apply(ctx, data);
    }

    public void Dispose()
    {
        _innerPipeline.Dispose();
    }

    public void Reset()
    {
        _innerPipeline.Reset();
        _lastLogits = null;
        _lastContext = null;
    }

    public LLamaToken Sample(SafeLLamaContextHandle ctx, int index)
    {
        // Get the logits for this position BEFORE sampling
        var logits = ctx.GetLogitsIth(index);

        // Copy logits for probability calculation (sampling may modify them)
        _lastLogits = logits.ToArray();
        _lastContext = ctx;

        // Calculate softmax probabilities BEFORE sampling
        var probabilities = CalculateSoftmax(_lastLogits);

        // Get top candidates before sampling
        var topCandidates = GetTopCandidates(_lastLogits, probabilities, ctx, 5);

        // Calculate entropy from probability distribution
        var entropy = CalculateEntropy(probabilities);

        // Let the inner pipeline do the actual sampling
        var sampledToken = _innerPipeline.Sample(ctx, index);

        // Get the probability of the sampled token
        var tokenId = (int)sampledToken;
        var probability = tokenId >= 0 && tokenId < probabilities.Length ? probabilities[tokenId] : 0f;
        var logProb = probability > 0 ? MathF.Log(probability) : float.NegativeInfinity;

        // Get token text
        var tokenText = GetTokenText(ctx, sampledToken);

        // Create token info
        LastTokenInfo = new TokenProbabilityInfo
        {
            TokenId = tokenId,
            TokenText = tokenText,
            Probability = probability,
            LogProbability = logProb,
            Entropy = entropy,
            TopCandidates = topCandidates
        };

        // Notify callback
        _onTokenSampled?.Invoke(LastTokenInfo);

        return sampledToken;
    }

    private static float[] CalculateSoftmax(float[] logits)
    {
        // Find max for numerical stability
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
                maxLogit = logits[i];
        }

        // Calculate exp and sum
        var probs = new float[logits.Length];
        float sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - maxLogit);
            sum += probs[i];
        }

        // Normalize
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }

        return probs;
    }

    private static double CalculateEntropy(float[] probabilities)
    {
        double entropy = 0;
        foreach (var p in probabilities)
        {
            if (p > 1e-10f)
            {
                entropy -= p * Math.Log(p);
            }
        }
        return entropy;
    }

    private static List<TokenCandidate> GetTopCandidates(
        float[] logits,
        float[] probabilities,
        SafeLLamaContextHandle ctx,
        int topK)
    {
        // Create list of (index, probability) pairs
        var candidates = new List<(int index, float prob)>(logits.Length);
        for (int i = 0; i < probabilities.Length; i++)
        {
            candidates.Add((i, probabilities[i]));
        }

        // Sort by probability descending and take top K
        return candidates
            .OrderByDescending(c => c.prob)
            .Take(topK)
            .Select(c => new TokenCandidate
            {
                TokenId = c.index,
                TokenText = GetTokenTextById(ctx, c.index),
                Probability = c.prob,
                LogProbability = c.prob > 0 ? MathF.Log(c.prob) : float.NegativeInfinity
            })
            .ToList();
    }

    private static string GetTokenText(SafeLLamaContextHandle ctx, LLamaToken token)
    {
        try
        {
            var buffer = new byte[128];
            var model = ctx.ModelHandle;
            var length = model.TokenToSpan(token, buffer);
            return length > 0 ? System.Text.Encoding.UTF8.GetString(buffer, 0, (int)length) : $"[{(int)token}]";
        }
        catch
        {
            return $"[{(int)token}]";
        }
    }

    private static string GetTokenTextById(SafeLLamaContextHandle ctx, int tokenId)
    {
        try
        {
            var buffer = new byte[128];
            var model = ctx.ModelHandle;
            LLamaToken token = tokenId; // implicit conversion from int
            var length = model.TokenToSpan(token, buffer);
            return length > 0 ? System.Text.Encoding.UTF8.GetString(buffer, 0, (int)length) : $"[{tokenId}]";
        }
        catch
        {
            return $"[{tokenId}]";
        }
    }
}

/// <summary>
/// Information about a sampled token including its probability
/// </summary>
public class TokenProbabilityInfo
{
    public int TokenId { get; set; }
    public string TokenText { get; set; } = string.Empty;
    public float Probability { get; set; }
    public float LogProbability { get; set; }
    public double Entropy { get; set; }
    public List<TokenCandidate> TopCandidates { get; set; } = new();
}
