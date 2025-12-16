using System.Collections;
using System.Reflection;
using LLama;
using LLama.Common;

namespace LocalLLMChat.Rag;

/// <summary>
/// GGUF-only embedding generator using LLama.LLamaEmbedder (reflection-based).
/// </summary>
public class GgufEmbeddingGenerator(RagConfig config, ILogger<GgufEmbeddingGenerator> logger) : IDisposable
{
    private readonly RagConfig _config = config;
    private readonly ILogger<GgufEmbeddingGenerator> _logger = logger;
    private readonly object _lock = new();

    private LLamaWeights? _weights;
    private LLamaContext? _context;
    private object? _embedder;
    private MethodInfo? _embedMethod;
    private bool _disposed;

    public Task<IReadOnlyList<float>> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        if (text == null) text = string.Empty;
        text = text.Trim();
        if (text.Length == 0)
        {
            return Task.FromResult<IReadOnlyList<float>>(Array.Empty<float>());
        }

        lock (_lock)
        {
            EnsureInitialized();
            var vector = InvokeEmbedder(text, cancellationToken);
            return Task.FromResult<IReadOnlyList<float>>(vector);
        }
    }

    public Task<IReadOnlyList<IReadOnlyList<float>>> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        if (texts == null || texts.Count == 0)
        {
            return Task.FromResult<IReadOnlyList<IReadOnlyList<float>>>(Array.Empty<IReadOnlyList<float>>());
        }

        lock (_lock)
        {
            EnsureInitialized();
            var results = new List<IReadOnlyList<float>>(texts.Count);
            foreach (var t in texts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var text = (t ?? string.Empty).Trim();
                if (text.Length == 0)
                {
                    results.Add(Array.Empty<float>());
                    continue;
                }
                results.Add(InvokeEmbedder(text, cancellationToken));
            }

            return Task.FromResult<IReadOnlyList<IReadOnlyList<float>>>(results);
        }
    }

    private void EnsureInitialized()
    {
        if (_embedder != null && _embedMethod != null) return;

        var modelPath = ResolvePath(_config.EmbeddingModelPath);
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Embedding GGUF not found at '{modelPath}'. Place the embedding model there.");
        }

        var modelParams = BuildModelParams(modelPath);

        _weights = LoadWeights(modelParams);
        _context = _weights.CreateContext(modelParams);

        var embedderType = typeof(LLamaWeights).Assembly.GetType("LLama.LLamaEmbedder");
        if (embedderType == null)
        {
            throw new InvalidOperationException("LLama.LLamaEmbedder not found. Install a LLamaSharp backend that includes embedding support.");
        }

        var embedParams = CreateEmbeddingParams();

        foreach (var ctor in embedderType.GetConstructors())
        {
            var args = TryBuildCtorArgs(ctor, modelParams, _weights, _context, embedParams);
            if (args == null) continue;
            try
            {
                _embedder = ctor.Invoke(args);
                if (_embedder != null) break;
            }
            catch
            {
                // try next ctor
            }
        }

        if (_embedder == null)
        {
            throw new InvalidOperationException("Failed to initialize LLama.LLamaEmbedder. Ensure your LLamaSharp package supports embeddings.");
        }

        _embedMethod = FindEmbedMethod(_embedder.GetType());
        if (_embedMethod == null)
        {
            throw new InvalidOperationException("No embedding method found on LLama.LLamaEmbedder. Update LLamaSharp to a version that exposes embedding APIs.");
        }
    }

    private static ModelParams BuildModelParams(string modelPath)
    {
        var mp = new ModelParams(modelPath);
        // Keep context modest for embeddings; ensure CPU-only for safety.
        SafeSet(mp, "ContextSize", (uint)2048);
        SafeSet(mp, "GpuLayerCount", 0);
        return mp;
    }

    private static object[]? TryBuildCtorArgs(ConstructorInfo ctor, ModelParams mp, LLamaWeights? weights, LLamaContext? ctx, object? embedParams)
    {
        var parms = ctor.GetParameters();
        var args = new object?[parms.Length];

        for (int i = 0; i < parms.Length; i++)
        {
            var pType = parms[i].ParameterType;
            if (pType == typeof(ModelParams) || pType.IsAssignableFrom(mp.GetType()))
            {
                args[i] = mp;
            }
            else if (pType == typeof(LLamaWeights) || (weights != null && pType.IsInstanceOfType(weights)))
            {
                if (weights == null) return null;
                args[i] = weights;
            }
            else if (pType == typeof(LLamaContext) || (ctx != null && pType.IsInstanceOfType(ctx)))
            {
                if (ctx == null) return null;
                args[i] = ctx;
            }
            else if (embedParams != null && pType.IsAssignableFrom(embedParams.GetType()))
            {
                args[i] = embedParams;
            }
            else
            {
                return null;
            }
        }

        return args;
    }

    private static MethodInfo? FindEmbedMethod(Type type)
    {
        // Prefer single-string methods named with "Embed" or "Embedding"
        var methods = type.GetMethods(BindingFlags.Instance | BindingFlags.Public);
        foreach (var m in methods)
        {
            if (!m.Name.Contains("Embed", StringComparison.OrdinalIgnoreCase)) continue;
            var ps = m.GetParameters();
            if (ps.Length == 1 && ps[0].ParameterType == typeof(string) && m.ReturnType != typeof(void))
            {
                return m;
            }
            if (ps.Length == 2 && ps[0].ParameterType == typeof(string) && ps[1].ParameterType == typeof(CancellationToken))
            {
                return m;
            }
        }

        // Fallback: first public method with string param and non-void return
        foreach (var m in methods)
        {
            var ps = m.GetParameters();
            if (ps.Length >= 1 && ps[0].ParameterType == typeof(string) && m.ReturnType != typeof(void))
            {
                return m;
            }
        }

        return null;
    }

    private IReadOnlyList<float> InvokeEmbedder(string text, CancellationToken cancellationToken)
    {
        if (_embedder == null || _embedMethod == null)
        {
            throw new InvalidOperationException("Embedder is not initialized.");
        }

        var ps = _embedMethod.GetParameters();
        var args = new object?[ps.Length];
        for (int i = 0; i < ps.Length; i++)
        {
            var p = ps[i];
            if (p.ParameterType == typeof(string))
            {
                args[i] = text;
            }
            else if (p.ParameterType == typeof(CancellationToken))
            {
                args[i] = cancellationToken;
            }
            else if (p.HasDefaultValue)
            {
                args[i] = p.DefaultValue ?? Type.Missing;
            }
            else
            {
                // unsupported parameter type
                throw new InvalidOperationException($"Embedding method {_embedMethod.Name} has unsupported parameter type {p.ParameterType.Name}.");
            }
        }

        var result = _embedMethod.Invoke(_embedder, args);
        return ToFloatArray(result);
    }

    private static IReadOnlyList<float> ToFloatArray(object? value)
    {
        if (value == null) return Array.Empty<float>();
        if (value is float[] arr) return arr;
        if (value is IReadOnlyList<float> roList) return roList;
        if (value is IEnumerable<float> enumerableFloats) return enumerableFloats.ToArray();

        if (value is IEnumerable enumerable)
        {
            var list = new List<float>();
            foreach (var item in enumerable)
            {
                if (item == null) continue;
                list.Add(Convert.ToSingle(item));
            }
            return list.ToArray();
        }

        // Handle ReadOnlyMemory<float> / ReadOnlySpan<float> via reflection
        var type = value.GetType();
        if (type.FullName?.StartsWith("System.ReadOnlyMemory") == true || type.FullName?.StartsWith("System.Memory") == true)
        {
            var spanProp = type.GetProperty("Span");
            var spanObj = spanProp?.GetValue(value);
            return ToFloatArray(spanObj);
        }

        if (type.FullName?.StartsWith("System.ReadOnlySpan") == true || type.FullName?.StartsWith("System.Span") == true)
        {
            // Copy via IEnumerable path if available
            var toArray = type.GetMethod("ToArray");
            if (toArray != null)
            {
                var arrObj = toArray.Invoke(value, Array.Empty<object>());
                return ToFloatArray(arrObj);
            }
        }

        throw new InvalidOperationException($"Unable to convert embedding result of type {type.FullName} to float array.");
    }

    private static object? CreateEmbeddingParams()
    {
        var type = Type.GetType("LLama.Common.EmbeddingParams, LLama") ?? Type.GetType("LLama.EmbeddingParams, LLama");
        if (type == null) return null;
        try
        {
            return Activator.CreateInstance(type);
        }
        catch
        {
            return null;
        }
    }

    private static LLamaWeights LoadWeights(ModelParams modelParams)
    {
        try
        {
            var loadAsync = typeof(LLamaWeights).GetMethod("LoadFromFileAsync", BindingFlags.Public | BindingFlags.Static);
            if (loadAsync != null)
            {
                var args = BuildWeightLoadArgs(loadAsync.GetParameters(), modelParams);
                if (args != null)
                {
                    var taskObj = loadAsync.Invoke(null, args);
                    if (taskObj is Task<LLamaWeights> tWeights)
                    {
                        return tWeights.GetAwaiter().GetResult();
                    }
                    if (taskObj is Task t)
                    {
                        t.GetAwaiter().GetResult();
                        var result = t.GetType().GetProperty("Result")?.GetValue(t) as LLamaWeights;
                        if (result != null) return result;
                    }
                }
            }

            var load = typeof(LLamaWeights).GetMethod("LoadFromFile", BindingFlags.Public | BindingFlags.Static);
            if (load != null)
            {
                var args = BuildWeightLoadArgs(load.GetParameters(), modelParams);
                if (args != null)
                {
                    var weightsObj = load.Invoke(null, args);
                    if (weightsObj is LLamaWeights lw) return lw;
                }
            }
        }
        catch (Exception e)
        {
            throw new InvalidOperationException($"Unable to load embedding weights from GGUF. {e}");
        }

        throw new InvalidOperationException("Unable to load embedding weights from GGUF. Verify the GGUF path and LLamaSharp version.");
    }

    private static object?[]? BuildWeightLoadArgs(ParameterInfo[] parameters, ModelParams modelParams)
    {
        // Adapt to multiple LLamaSharp signatures (sync/async, optional CT/progress)
        var args = new object?[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var pType = parameters[i].ParameterType;
            if (pType.IsAssignableFrom(modelParams.GetType()))
            {
                args[i] = modelParams;
            }
            else if (pType == typeof(CancellationToken))
            {
                args[i] = CancellationToken.None;
            }
            else if (typeof(IProgress<float>).IsAssignableFrom(pType))
            {
                args[i] = null;
            }
            else if (pType == typeof(string))
            {
                args[i] = modelParams.ModelPath;
            }
            else
            {
                return null;
            }
        }

        return args;
    }

    private static void SafeSet(object target, string propertyName, object value)
    {
        var prop = target.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);
        if (prop == null || !prop.CanWrite) return;
        try
        {
            prop.SetValue(target, Convert.ChangeType(value, Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType));
        }
        catch
        {
            // ignore
        }
    }

    private static string ResolvePath(string path)
    {
        if (path.StartsWith("~"))
        {
            var root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
            return Path.Combine(root, path.TrimStart('~', '/', '\\'));
        }
        return path;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        (_embedder as IDisposable)?.Dispose();
        _context?.Dispose();
        _weights?.Dispose();
    }
}
