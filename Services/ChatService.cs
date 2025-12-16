using LocalLLMChat.Models;
using LocalLLMChat.Services.Plugins;
using LocalLLMChat.Rag;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace LocalLLMChat.Services;

public class ChatService : IChatService
{
    private readonly ILlmService _llmService;
    private readonly Rag.RagPipeline _ragPipeline;
    private readonly IEnumerable<IChatPlugin> _plugins;
    private readonly ILogger<ChatService> _logger;
    private readonly IServerDebugLog _debugLog;
    private readonly ConcurrentDictionary<string, ChatSession> _sessions = new();

    public ChatService(
        ILlmService llmService,
        Rag.RagPipeline ragPipeline,
        IEnumerable<IChatPlugin> plugins,
        IServerDebugLog debugLog,
        ILogger<ChatService> logger)
    {
        _llmService = llmService;
        _ragPipeline = ragPipeline;
        _plugins = plugins.OrderBy(p => p.Priority).ToList();
        _debugLog = debugLog;
        _logger = logger;
    }

    public ChatSession CreateSession()
    {
        var session = new ChatSession();
        _sessions[session.SessionId] = session;
        _logger.LogInformation("Created new session: {SessionId}", session.SessionId);
        return session;
    }

    public ChatSession? GetSession(string sessionId)
    {
        _sessions.TryGetValue(sessionId, out var session);
        return session;
    }

    public void ClearSession(string sessionId)
    {
        if (_sessions.TryRemove(sessionId, out _))
        {
            _logger.LogInformation("Cleared session: {SessionId}", sessionId);
        }
    }

    public async Task<ChatResponse> ProcessMessageAsync(ChatRequest request, CancellationToken cancellationToken = default)
    {
        try
        {
            _debugLog.Add("chat", $"ProcessMessage (stream=false), msg len={request.Message?.Length ?? 0}");

            // Get or create session
            var session = string.IsNullOrEmpty(request.SessionId)
                ? CreateSession()
                : GetSession(request.SessionId) ?? CreateSession();

            // Create plugin context
            var context = new PluginContext
            {
                UserMessage = request.Message,
                ProcessedPrompt = request.Message,
                Session = session
            };

            // Run pre-inference plugins
            context = await RunPreInferencePluginsAsync(context, cancellationToken);

            if (context.StopProcessing)
            {
                return new ChatResponse
                {
                    Response = context.Response,
                    SessionId = session.SessionId,
                    Success = true
                };
            }

            if (request.UseRag)
            {
                _debugLog.Add("chat", "RAG mode active (sync)");
                var ragResult = await _ragPipeline.AskAsync(context.ProcessedPrompt, cancellationToken);
                var sourcesText = ragResult.Sources.Any()
                    ? "\n\nSources:\n" + string.Join("\n", ragResult.Sources.Select(s => $"{s.SourceId} #{s.ChunkIndex} (score {s.Score:F3})"))
                    : string.Empty;
                context.Response = ragResult.Answer + sourcesText;
            }
            else
            {
                // Generate response from LLM
                _logger.LogDebug("Sending prompt to LLM: {Prompt}", context.ProcessedPrompt);
                _debugLog.Add("chat", $"LLM request len={context.ProcessedPrompt.Length}");
                context.Response = await _llmService.GenerateResponseAsync(context.ProcessedPrompt, cancellationToken);
            }

            // Run post-inference plugins
            context = await RunPostInferencePluginsAsync(context, cancellationToken);

            _debugLog.Add("chat", $"LLM response len={context.Response.Length}");
            return new ChatResponse
            {
                Response = context.Response,
                SessionId = session.SessionId,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing message");
            _debugLog.Add("chat", $"Error: {ex.Message}");
            return new ChatResponse
            {
                Response = string.Empty,
                SessionId = request.SessionId ?? string.Empty,
                Success = false,
                Error = ex.Message
            };
        }
    }

    public async IAsyncEnumerable<string> ProcessMessageStreamingAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        _debugLog.Add("chat", $"ProcessMessage (stream=true), msg len={request.Message?.Length ?? 0}");

        // Get or create session
        var session = string.IsNullOrEmpty(request.SessionId)
            ? CreateSession()
            : GetSession(request.SessionId) ?? CreateSession();

        // Create plugin context
        var context = new PluginContext
        {
            UserMessage = request.Message,
            ProcessedPrompt = request.Message,
            Session = session
        };

        // Run pre-inference plugins
        context = await RunPreInferencePluginsAsync(context, cancellationToken);

        if (context.StopProcessing)
        {
            yield return context.Response;
            yield break;
        }

        if (request.UseRag)
        {
            _debugLog.Add("chat", "RAG mode active (stream)");
            var result = await _ragPipeline.AskAsync(context.ProcessedPrompt, cancellationToken);
            var sourcesText = result.Sources.Any()
                ? "\n\nSources:\n" + string.Join("\n", result.Sources.Select(s => $"{s.SourceId} #{s.ChunkIndex} (score {s.Score:F3})"))
                : string.Empty;
            yield return result.Answer + sourcesText;
            yield break;
        }
        else
        {
            // Stream response from LLM
            var fullResponse = new System.Text.StringBuilder();

            await foreach (var token in _llmService.GenerateStreamingResponseAsync(context.ProcessedPrompt, cancellationToken))
            {
                fullResponse.Append(token);
                yield return token;
            }

            // Update context with full response for post-processing
            context.Response = fullResponse.ToString();

            // Run post-inference plugins (for history, etc.)
            await RunPostInferencePluginsAsync(context, cancellationToken);

            _debugLog.Add("chat", $"Stream complete, len={context.Response.Length}");
        }
    }

    private async Task<PluginContext> RunPreInferencePluginsAsync(PluginContext context, CancellationToken cancellationToken)
    {
        foreach (var plugin in _plugins.Where(p => p.IsEnabled))
        {
            _logger.LogDebug("Running pre-inference plugin: {PluginName}", plugin.Name);
            context = await plugin.ProcessBeforeInferenceAsync(context, cancellationToken);

            if (context.StopProcessing)
            {
                _logger.LogInformation("Plugin {PluginName} stopped processing", plugin.Name);
                break;
            }
        }

        return context;
    }

    private async Task<PluginContext> RunPostInferencePluginsAsync(PluginContext context, CancellationToken cancellationToken)
    {
        foreach (var plugin in _plugins.Where(p => p.IsEnabled))
        {
            _logger.LogDebug("Running post-inference plugin: {PluginName}", plugin.Name);
            context = await plugin.ProcessAfterInferenceAsync(context, cancellationToken);

            if (context.StopProcessing) break;
        }

        return context;
    }
}
