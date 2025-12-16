using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using LocalLLMChat.Models;

namespace LocalLLMChat.Services;

public class LlamaSharpService : ILlmService, IDisposable
{
    private readonly ILlmSettingsStore _settingsStore;
    private readonly ILogger<LlamaSharpService> _logger;
    private readonly IDiagnosticsService _diagnostics;
    private readonly IServerDebugLog _debugLog;
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly SemaphoreSlim _inferenceLock = new(1, 1);
    private LLamaWeights? _model;
    private LLamaContext? _context;
    private bool _disposed;
    private LlmSettings _settings;
    private LlmSettings? _loadedSettingsSnapshot;
    private RuntimeLlmSettings _runtimeSettings;

    public bool IsModelLoaded => _model != null && _context != null;

    public LlmSettings Settings => _settings;
    public RuntimeLlmSettings RuntimeSettings => _runtimeSettings;

    public LlamaSharpService(
        ILlmSettingsStore settingsStore,
        ILogger<LlamaSharpService> logger,
        IDiagnosticsService diagnostics,
        IServerDebugLog debugLog)
    {
        _settingsStore = settingsStore;
        _logger = logger;
        _diagnostics = diagnostics;
        _debugLog = debugLog;

        _settings = NormalizeSettings(_settingsStore.Get());
        _runtimeSettings = CreateRuntimeSettingsFromConfig(_settings);
    }

    public async Task LoadModelAsync()
    {
        if (!_settings.KeepModelInMemory)
        {
            _logger.LogDebug("Skipping warm load because KeepModelInMemory is false.");
            return;
        }

        await EnsurePersistentModelLoadedAsync(_settings, CancellationToken.None);
    }

    public async Task<SettingsUpdateResult> ApplySettingsAsync(LlmSettings settings, CancellationToken cancellationToken = default)
    {
        await _inferenceLock.WaitAsync(cancellationToken);
        try
        {
            var normalized = NormalizeSettings(settings);
            var previous = _settings;

            _settings = normalized;
            _runtimeSettings = CreateRuntimeSettingsFromConfig(normalized);
            _settingsStore.Save(normalized);

            var requiresReload = RequiresReload(previous, normalized) || (!normalized.KeepModelInMemory && IsModelLoaded);

            if (requiresReload)
            {
                await ReloadModelUnsafeAsync(normalized, cancellationToken);
            }

            return new SettingsUpdateResult
            {
                Reloaded = requiresReload && normalized.KeepModelInMemory,
                Settings = normalized
            };
        }
        finally
        {
            _inferenceLock.Release();
        }
    }

    public void UpdateRuntimeSettings(RuntimeLlmSettings newSettings)
    {
        _runtimeSettings = newSettings;
        _logger.LogInformation("Runtime settings updated: Temp={Temp}, TopP={TopP}, TopK={TopK}, MaxTokens={MaxTokens}",
            newSettings.Temperature, newSettings.TopP, newSettings.TopK, newSettings.MaxTokens);
        _debugLog.Add("llm", $"Runtime settings updated (Temp={newSettings.Temperature}, TopP={newSettings.TopP}, TopK={newSettings.TopK})");
    }

    private RuntimeLlmSettings CreateRuntimeSettingsFromConfig(LlmSettings s)
    {
        return new RuntimeLlmSettings
        {
            MaxTokens = s.MaxTokens,
            Temperature = s.Temperature,
            TopP = s.TopP,
            TopK = s.TopK,
            MinP = s.MinP,
            RepeatPenalty = s.RepeatPenalty,
            RepeatLastN = s.RepeatLastN,
            FrequencyPenalty = s.FrequencyPenalty,
            PresencePenalty = s.PresencePenalty,
            Seed = s.RandomSeed ? -1 : (s.RuntimeSeed != -1 ? s.RuntimeSeed : s.Seed),
            SystemPrompt = s.SystemPrompt,
            StopSequences = new List<string>(s.StopSequences)
        };
    }

    private LlmSettings NormalizeSettings(LlmSettings settings)
    {
        var copy = settings.Clone();

        if (copy.CpuOnly)
        {
            copy.GpuOffloadLayers = 0;
            copy.OffloadKvCacheToGpu = false;
        }

        copy.ContextLength = Clamp(copy.ContextLength, 256, 262_144);
        copy.GpuOffloadLayers = Clamp(copy.GpuOffloadLayers, 0, 999);
        copy.CpuThreadPoolSize = Math.Clamp(copy.CpuThreadPoolSize, -1, 128);
        copy.EvaluationBatchSize = Clamp(copy.EvaluationBatchSize, 1, 4096);
        copy.Seed = Math.Max(0, copy.Seed);

        return copy;
    }

    private async Task EnsurePersistentModelLoadedAsync(LlmSettings settings, CancellationToken cancellationToken)
    {
        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            if (IsModelLoaded && _loadedSettingsSnapshot != null && !RequiresReload(_loadedSettingsSnapshot, settings))
            {
                return;
            }

            DisposeModel();
            var (weights, context) = await LoadModelArtifactsAsync(settings);
            _model = weights;
            _context = context;
            _loadedSettingsSnapshot = settings.Clone();
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private async Task ReloadModelUnsafeAsync(LlmSettings settings, CancellationToken cancellationToken)
    {
        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            DisposeModel();
            _loadedSettingsSnapshot = null;

            if (settings.KeepModelInMemory)
            {
                var (weights, context) = await LoadModelArtifactsAsync(settings);
                _model = weights;
                _context = context;
                _loadedSettingsSnapshot = settings.Clone();
            }
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private static bool RequiresReload(LlmSettings previous, LlmSettings updated)
    {
        return !string.Equals(previous.ModelPath, updated.ModelPath, StringComparison.OrdinalIgnoreCase)
            || previous.CpuOnly != updated.CpuOnly
            || previous.ContextLength != updated.ContextLength
            || previous.GpuOffloadLayers != updated.GpuOffloadLayers
            || previous.TryMmap != updated.TryMmap
            || previous.UseMlock != updated.UseMlock
            || previous.KeepModelInMemory != updated.KeepModelInMemory
            || previous.FlashAttention != updated.FlashAttention
            || previous.OffloadKvCacheToGpu != updated.OffloadKvCacheToGpu
            || previous.KCacheQuantTypeEnabled != updated.KCacheQuantTypeEnabled
            || !string.Equals(previous.KCacheQuantType, updated.KCacheQuantType, StringComparison.OrdinalIgnoreCase)
            || previous.VCacheQuantTypeEnabled != updated.VCacheQuantTypeEnabled
            || !string.Equals(previous.VCacheQuantType, updated.VCacheQuantType, StringComparison.OrdinalIgnoreCase)
            || previous.RopeFrequencyBaseEnabled != updated.RopeFrequencyBaseEnabled
            || previous.RopeFrequencyScaleEnabled != updated.RopeFrequencyScaleEnabled
            || FloatDiffers(previous.RopeFrequencyBase, updated.RopeFrequencyBase)
            || FloatDiffers(previous.RopeFrequencyScale, updated.RopeFrequencyScale);
    }

    private static bool FloatDiffers(float? a, float? b)
    {
        if (!a.HasValue && !b.HasValue) return false;
        if (!a.HasValue || !b.HasValue) return true;
        return Math.Abs(a.Value - b.Value) > 0.0001f;
    }

    private static int Clamp(int value, int min, int max) => Math.Min(Math.Max(value, min), max);

    private async Task<(LLamaWeights Weights, LLamaContext Context, bool IsTemporary)> GetOrCreateContextAsync(
        LlmSettings settings,
        CancellationToken cancellationToken)
    {
        if (settings.KeepModelInMemory)
        {
            await EnsurePersistentModelLoadedAsync(settings, cancellationToken);
            return (_model!, _context!, false);
        }

        await _loadLock.WaitAsync(cancellationToken);
        try
        {
            var (weights, context) = await LoadModelArtifactsAsync(settings);
            return (weights, context, true);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private async Task<(LLamaWeights Weights, LLamaContext Context)> LoadModelArtifactsAsync(LlmSettings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            throw new InvalidOperationException("ModelPath is empty. Please configure a model file.");
        }

        if (!File.Exists(settings.ModelPath))
        {
            _debugLog.Add("llm", $"Model file not found: {settings.ModelPath}");
            throw new FileNotFoundException($"Model file not found at: {settings.ModelPath}");
        }

        var fileName = Path.GetFileName(settings.ModelPath).ToUpperInvariant();
        if (fileName.Contains("MXFP") || fileName.Contains("MX4"))
        {
            throw new NotSupportedException(
                "MXFP4/MX4 quantization format is not supported by LlamaSharp. " +
                "Please use a different quantization like Q4_K_M, Q5_K_M, Q8_0, or F16.");
        }

        var modelParams = BuildModelParams(settings);

        _logger.LogInformation("Loading model from {ModelPath}", settings.ModelPath);
        _logger.LogInformation("Model params: Context={ContextLength}, GpuLayers={GpuLayers}, Batch={Batch}, Threads={Threads}, FlashAttn={FlashAttn}, mmap={Mmap}, mlock={Mlock}, kvOffload={KvOffload}",
            settings.ContextLength,
            settings.CpuOnly ? 0 : settings.GpuOffloadLayers,
            settings.EvaluationBatchSize,
            settings.CpuThreadPoolSize > 0 ? settings.CpuThreadPoolSize : -1,
            settings.FlashAttention,
            settings.TryMmap,
            settings.UseMlock,
            settings.OffloadKvCacheToGpu && !settings.CpuOnly);
        _debugLog.Add("llm", $"Loading model from {settings.ModelPath}");

        try
        {
            var weights = await LLamaWeights.LoadFromFileAsync(modelParams);
            var context = weights.CreateContext(modelParams);

            var modelSize = new FileInfo(settings.ModelPath).Length;
            var contextMemory = EstimateContextMemory(settings.ContextLength);
            _diagnostics.UpdateMemoryStats(modelSize, contextMemory);

            _debugLog.Add("llm", $"Model loaded, size={(modelSize / (1024.0 * 1024 * 1024)).ToString("F2")} GB, ctx={settings.ContextLength}");
            return (weights, context);
        }
        catch (Exception ex) when (ex.Message.Contains("GGML_ASSERT", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("type", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Failed to load model - unsupported quantization format. Error: {ex.Message}", ex);
        }
    }

    private static long EstimateContextMemory(int contextSize)
    {
        // Rough estimate: each token in context needs ~2KB for KV cache (varies by model)
        return (long)contextSize * 2048;
    }

    private ModelParams BuildModelParams(LlmSettings settings)
    {
        var modelParams = new ModelParams(settings.ModelPath);

        SetIntegralProperty(modelParams, "ContextSize", (uint)Math.Max(settings.ContextLength, 1));
        SetIntegralProperty(modelParams, "GpuLayerCount", settings.CpuOnly ? 0 : settings.GpuOffloadLayers);
        SetIntegralProperty(modelParams, "BatchSize", (uint)Math.Max(settings.EvaluationBatchSize, 1));

        var threadValue = settings.CpuThreadPoolSize > 0 ? settings.CpuThreadPoolSize : (int?)null;
        SetPropertyIfExists(modelParams, "Threads", threadValue);
        SetPropertyIfExists(modelParams, "BatchThreads", threadValue);

        SetPropertyIfExists(modelParams, "UseMemorymap", settings.TryMmap);
        SetPropertyIfExists(modelParams, "UseMemoryLock", settings.UseMlock);
        SetPropertyIfExists(modelParams, "FlashAttention", settings.FlashAttention);

        if (settings.RopeFrequencyBaseEnabled && settings.RopeFrequencyBase.HasValue)
        {
            SetFloatProperty(modelParams, "RopeFrequencyBase", settings.RopeFrequencyBase.Value);
        }

        if (settings.RopeFrequencyScaleEnabled && settings.RopeFrequencyScale.HasValue)
        {
            SetFloatProperty(modelParams, "RopeFrequencyScale", settings.RopeFrequencyScale.Value);
        }

        var noOffload = settings.CpuOnly || !settings.OffloadKvCacheToGpu;
        SetPropertyIfExists(modelParams, "NoKqvOffload", noOffload);

        if (settings.KCacheQuantTypeEnabled)
        {
            SetEnumPropertyIfExists(modelParams, "TypeK", settings.KCacheQuantType);
        }

        if (settings.VCacheQuantTypeEnabled)
        {
            SetEnumPropertyIfExists(modelParams, "TypeV", settings.VCacheQuantType);
        }

        var seedValue = settings.RandomSeed ? (int)GenerateSeed() : Math.Max(0, settings.Seed);
        SetSeedProperty(modelParams, seedValue);

        return modelParams;
    }

    private static uint GenerateSeed()
    {
        return (uint)RandomNumberGenerator.GetInt32(int.MaxValue);
    }

    private static void SetIntegralProperty(object target, string propertyName, object value)
    {
        SetPropertyIfExists(target, propertyName, value);
    }

    private static void SetFloatProperty(object target, string propertyName, float value)
    {
        SetPropertyIfExists(target, propertyName, value);
    }

    private static void SetSeedProperty(object target, int seed)
    {
        var property = target.GetType().GetProperty("Seed", BindingFlags.Instance | BindingFlags.Public);
        if (property == null || !property.CanWrite) return;

        try
        {
            var destinationType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
            object converted = destinationType == typeof(uint)
                ? (uint)seed
                : Convert.ChangeType(seed, destinationType);
            property.SetValue(target, converted);
        }
        catch
        {
            // Ignore unsupported seed shapes
        }
    }

    private static void SetEnumPropertyIfExists(object target, string propertyName, string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return;

        var property = target.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);
        if (property == null || !property.CanWrite) return;

        var enumType = property.PropertyType;
        if (!enumType.IsEnum && Nullable.GetUnderlyingType(enumType)?.IsEnum == true)
        {
            enumType = Nullable.GetUnderlyingType(enumType)!;
        }

        if (!enumType.IsEnum) return;

        try
        {
            var parsed = Enum.Parse(enumType, value, ignoreCase: true);
            property.SetValue(target, parsed);
        }
        catch
        {
            // Ignore unsupported enum values
        }
    }

    private static void SetPropertyIfExists(object target, string propertyName, object? value)
    {
        var property = target.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);
        if (property == null || !property.CanWrite) return;

        var destinationType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;

        try
        {
            if (value == null && destinationType.IsValueType && Nullable.GetUnderlyingType(property.PropertyType) == null)
            {
                return;
            }

            if (value != null && !destinationType.IsAssignableFrom(value.GetType()))
            {
                value = Convert.ChangeType(value, destinationType);
            }

            property.SetValue(target, value);
        }
        catch
        {
            // Ignore unsupported properties/types
        }
    }

    private DefaultSamplingPipeline CreateBaseSamplingPipeline()
    {
        var pipeline = new DefaultSamplingPipeline
        {
            Temperature = _runtimeSettings.Temperature,
            TopP = _runtimeSettings.TopP,
            TopK = _runtimeSettings.TopK,
            MinP = _runtimeSettings.MinP,
            RepeatPenalty = _runtimeSettings.RepeatPenalty,
            PenaltyCount = _runtimeSettings.RepeatLastN,
            FrequencyPenalty = _runtimeSettings.FrequencyPenalty,
            PresencePenalty = _runtimeSettings.PresencePenalty
        };

        if (_runtimeSettings.Seed >= 0)
        {
            pipeline.Seed = (uint)_runtimeSettings.Seed;
        }

        return pipeline;
    }

    private DiagnosticSamplingPipeline CreateDiagnosticSamplingPipeline(Action<TokenProbabilityInfo> onTokenSampled)
    {
        var basePipeline = CreateBaseSamplingPipeline();
        return new DiagnosticSamplingPipeline(basePipeline, _diagnostics, onTokenSampled);
    }

    private InferenceParams CreateInferenceParams(ISamplingPipeline samplingPipeline)
    {
        return new InferenceParams
        {
            MaxTokens = _runtimeSettings.MaxTokens,
            SamplingPipeline = samplingPipeline,
            AntiPrompts = _runtimeSettings.StopSequences
        };
    }

    private static int EstimateTokenCount(string text)
    {
        // Rough estimate: ~4 characters per token for English text
        return Math.Max(1, text.Length / 4);
    }

    public async Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
    {
        _debugLog.Add("llm", $"Generate (sync) prompt len={prompt?.Length ?? 0}");

        await _inferenceLock.WaitAsync(cancellationToken);
        DiagnosticSamplingPipeline? diagnosticPipeline = null;
        LLamaWeights? tempWeights = null;
        LLamaContext? tempContext = null;

        try
        {
            var settingsSnapshot = _settings.Clone();
            var contextHandle = await GetOrCreateContextAsync(settingsSnapshot, cancellationToken);
            tempWeights = contextHandle.Weights;
            tempContext = contextHandle.Context;

            ResetContextForRequest(contextHandle.Context, contextHandle.IsTemporary);

            var promptTokens = EstimateTokenCount(prompt);
            _diagnostics.StartInference(promptTokens, _runtimeSettings, settingsSnapshot.ContextSize);

            var executor = new InteractiveExecutor(contextHandle.Context);

            diagnosticPipeline = CreateDiagnosticSamplingPipeline(tokenInfo =>
            {
                _diagnostics.RecordToken(
                    tokenId: tokenInfo.TokenId,
                    tokenText: tokenInfo.TokenText,
                    probability: tokenInfo.Probability,
                    topCandidates: tokenInfo.TopCandidates
                );
            });

            var inferenceParams = CreateInferenceParams(diagnosticPipeline);

            var result = new System.Text.StringBuilder();

            await foreach (var token in executor.InferAsync(prompt, inferenceParams, cancellationToken))
            {
                result.Append(token);
            }

            _diagnostics.EndInference();

            _debugLog.Add("llm", $"Generate complete, len={result.Length}");
            return result.ToString().Trim();
        }
        finally
        {
            diagnosticPipeline?.Dispose();

            if (tempContext != null && tempWeights != null && tempWeights != _model)
            {
                tempContext.Dispose();
                tempWeights.Dispose();
            }

            _inferenceLock.Release();
        }
    }

    public async IAsyncEnumerable<string> GenerateStreamingResponseAsync(
        string prompt,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        _debugLog.Add("llm", $"Generate (stream) prompt len={prompt?.Length ?? 0}");

        await _inferenceLock.WaitAsync(cancellationToken);
        DiagnosticSamplingPipeline? diagnosticPipeline = null;
        LLamaWeights? tempWeights = null;
        LLamaContext? tempContext = null;

        try
        {
            var settingsSnapshot = _settings.Clone();
            var contextHandle = await GetOrCreateContextAsync(settingsSnapshot, cancellationToken);
            tempWeights = contextHandle.Weights;
            tempContext = contextHandle.Context;

            ResetContextForRequest(contextHandle.Context, contextHandle.IsTemporary);

            var promptTokens = EstimateTokenCount(prompt);
            _diagnostics.StartInference(promptTokens, _runtimeSettings, settingsSnapshot.ContextSize);

            var executor = new InteractiveExecutor(contextHandle.Context);

            diagnosticPipeline = CreateDiagnosticSamplingPipeline(tokenInfo =>
            {
                _diagnostics.RecordToken(
                    tokenId: tokenInfo.TokenId,
                    tokenText: tokenInfo.TokenText,
                    probability: tokenInfo.Probability,
                    topCandidates: tokenInfo.TopCandidates
                );
            });

            var inferenceParams = CreateInferenceParams(diagnosticPipeline);

            await foreach (var token in executor.InferAsync(prompt, inferenceParams, cancellationToken))
            {
                yield return token;
            }

            _diagnostics.EndInference();
        }
        finally
        {
            diagnosticPipeline?.Dispose();

            if (tempContext != null && tempWeights != null && tempWeights != _model)
            {
                tempContext.Dispose();
                tempWeights.Dispose();
            }

            _inferenceLock.Release();
        }

        _debugLog.Add("llm", "Stream inference completed");
    }

    private void ResetContextForRequest(LLamaContext context, bool isTemporary)
    {
        // If reusing a long-lived context, clear KV/cache between prompts to avoid invalid batches
        if (context == null || isTemporary) return;

        try
        {
            context.NativeHandle.MemoryClear(keepAllocations: true);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to reset LLama context before inference");
            _debugLog.Add("llm", $"Context reset failed: {ex.Message}");
        }
    }

    private void DisposeModel()
    {
        _context?.Dispose();
        _model?.Dispose();
        _context = null;
        _model = null;
        _loadedSettingsSnapshot = null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        DisposeModel();
        _loadLock.Dispose();
        _inferenceLock.Dispose();
    }
}
