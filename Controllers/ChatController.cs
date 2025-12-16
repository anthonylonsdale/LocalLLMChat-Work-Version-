using LocalLLMChat.Models;
using LocalLLMChat.Services;
using Microsoft.AspNetCore.Mvc;

namespace LocalLLMChat.Controllers;

public class ChatController(
    IChatService chatService,
    IRagService ragService,
    ILlmService llmService,
    IDiagnosticsService diagnosticsService,
    IServerDebugLog debugLog,
    ILogger<ChatController> logger) : Controller
{
    private readonly IChatService _chatService = chatService;
    private readonly IRagService _ragService = ragService;
    private readonly ILlmService _llmService = llmService;
    private readonly IDiagnosticsService _diagnosticsService = diagnosticsService;
    private readonly IServerDebugLog _debugLog = debugLog;
    private readonly ILogger<ChatController> _logger = logger;

    public IActionResult Index()
    {
        var session = _chatService.CreateSession();
        ViewBag.SessionId = session.SessionId;
        return View();
    }

    [HttpPost]
    public async Task<IActionResult> Send([FromBody] ChatRequest request, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Message))
        {
            return BadRequest(new { error = "Message cannot be empty" });
        }

        _debugLog.Add("api", $"Send message len={request.Message.Length}");
        var response = await _chatService.ProcessMessageAsync(request, cancellationToken);

        if (!response.Success)
        {
            _debugLog.Add("api", $"Send error: {response.Error}");
            return StatusCode(500, new { error = response.Error });
        }

        return Json(response);
    }

    [HttpGet]
    public async Task SendStream([FromQuery] string message, [FromQuery] string? sessionId, [FromQuery] bool useRag, CancellationToken cancellationToken)
    {
        Response.ContentType = "text/event-stream";
        Response.Headers.Append("Cache-Control", "no-cache");
        Response.Headers.Append("Connection", "keep-alive");

        var request = new ChatRequest
        {
            Message = message,
            SessionId = sessionId,
            UseRag = useRag
        };

        try
        {
            _debugLog.Add("api", $"SendStream start len={message?.Length ?? 0}");
            await foreach (var token in _chatService.ProcessMessageStreamingAsync(request, cancellationToken))
            {
                await Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(new { token })}\n\n", cancellationToken);
                await Response.Body.FlushAsync(cancellationToken);
            }

            await Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Stream cancelled by client");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during streaming");
            _debugLog.Add("api", $"SendStream error: {ex.Message}");
            await Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(new { error = ex.Message })}\n\n", cancellationToken);
        }
    }

    [HttpPost]
    public IActionResult ClearSession([FromBody] string sessionId)
    {
        _chatService.ClearSession(sessionId);
        return Ok();
    }

    [HttpGet]
    public async Task<IActionResult> RagStats()
    {
        var count = await _ragService.GetDocumentCountAsync();
        return Json(new { documentCount = count });
    }

    [HttpPost]
    public async Task<IActionResult> AddDocument([FromBody] AddDocumentRequest request, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Content))
        {
            return BadRequest(new { error = "Content cannot be empty" });
        }

        _debugLog.Add("api", "AddDocument called");
        await _ragService.AddDocumentAsync(request.Content, request.Metadata, cancellationToken);
        return Ok(new { success = true });
    }

    // Settings API endpoints
    [HttpGet]
    public IActionResult GetSettings()
    {
        return Json(new
        {
            modelSettings = new
            {
                modelPath = _llmService.Settings.ModelPath,
                contextSize = _llmService.Settings.ContextSize,
                gpuLayerCount = _llmService.Settings.GpuLayerCount,
                batchSize = _llmService.Settings.BatchSize,
                threadCount = _llmService.Settings.ThreadCount,
                useFlashAttention = _llmService.Settings.UseFlashAttention
            },
            runtimeSettings = _llmService.RuntimeSettings,
            isModelLoaded = _llmService.IsModelLoaded
        });
    }

    [HttpPost]
    public IActionResult UpdateSettings([FromBody] RuntimeLlmSettings settings)
    {
        try
        {
            _llmService.UpdateRuntimeSettings(settings);
            return Json(new { success = true, settings = _llmService.RuntimeSettings });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating settings");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    [HttpPost]
    public IActionResult ResetSettings()
    {
        try
        {
            var defaultSettings = new RuntimeLlmSettings
            {
                MaxTokens = _llmService.Settings.MaxTokens,
                Temperature = _llmService.Settings.Temperature,
                TopP = _llmService.Settings.TopP,
                TopK = _llmService.Settings.TopK,
                MinP = _llmService.Settings.MinP,
                RepeatPenalty = _llmService.Settings.RepeatPenalty,
                RepeatLastN = _llmService.Settings.RepeatLastN,
                FrequencyPenalty = _llmService.Settings.FrequencyPenalty,
                PresencePenalty = _llmService.Settings.PresencePenalty,
                Seed = _llmService.Settings.Seed,
                SystemPrompt = _llmService.Settings.SystemPrompt,
                StopSequences = new List<string>(_llmService.Settings.StopSequences)
            };

            _llmService.UpdateRuntimeSettings(defaultSettings);
            return Json(new { success = true, settings = _llmService.RuntimeSettings });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error resetting settings");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    // Diagnostics API endpoints
    [HttpGet]
    public IActionResult GetDiagnostics()
    {
        var diagnostics = _diagnosticsService.CurrentDiagnostics;
        return Json(new
        {
            // Timing
            isGenerating = diagnostics.IsGenerating,
            status = diagnostics.Status,
            timeToFirstTokenMs = Math.Round(diagnostics.TimeToFirstTokenMs, 1),
            totalInferenceTimeMs = Math.Round(diagnostics.TotalInferenceTimeMs, 1),

            // Token Stats
            promptTokenCount = diagnostics.PromptTokenCount,
            generatedTokenCount = diagnostics.GeneratedTokenCount,
            tokensPerSecond = Math.Round(diagnostics.TokensPerSecond, 2),
            promptProcessingSpeed = Math.Round(diagnostics.PromptProcessingSpeed, 2),

            // Context
            contextSize = diagnostics.ContextSize,
            contextUsed = diagnostics.ContextUsed,
            contextUtilization = Math.Round(diagnostics.ContextUtilization, 1),

            // Sampling Info
            temperature = diagnostics.Temperature,
            topP = diagnostics.TopP,
            topK = diagnostics.TopK,
            minP = diagnostics.MinP,
            repeatPenalty = diagnostics.RepeatPenalty,

            // Memory
            modelMemory = diagnostics.ModelMemoryFormatted,
            contextMemory = diagnostics.ContextMemoryFormatted,

            // Stats
            averageEntropy = Math.Round(diagnostics.AverageTokenEntropy, 4),
            estimatedPerplexity = Math.Round(diagnostics.EstimatedPerplexity, 2),

            // Recent tokens (last 10)
            recentTokens = diagnostics.RecentTokens.TakeLast(10).Select(t => new
            {
                tokenId = t.TokenId,
                text = t.TokenText,
                probability = Math.Round(t.Probability, 6),
                logProbability = Math.Round(t.LogProbability, 4),
                entropy = Math.Round(t.Entropy, 4),
                generationTimeMs = Math.Round(t.GenerationTimeMs, 1),
                topCandidates = t.TopCandidates.Take(5).Select(c => new
                {
                    tokenId = c.TokenId,
                    text = c.TokenText,
                    probability = Math.Round(c.Probability, 6),
                    logProbability = Math.Round(c.LogProbability, 4)
                })
            }),

            // Last token's top candidates for detailed view
            lastTokenCandidates = diagnostics.RecentTokens.LastOrDefault()?.TopCandidates.Take(5).Select(c => new
            {
                tokenId = c.TokenId,
                text = c.TokenText,
                probability = Math.Round(c.Probability, 6),
                percentDisplay = (c.Probability * 100).ToString("F2") + "%"
            }) ?? Enumerable.Empty<object>()
        });
    }

    [HttpGet]
    public async Task StreamDiagnostics(CancellationToken cancellationToken)
    {
        Response.ContentType = "text/event-stream";
        Response.Headers.Append("Cache-Control", "no-cache");
        Response.Headers.Append("Connection", "keep-alive");

        var tcs = new TaskCompletionSource<bool>();

        void OnDiagnosticsUpdated(object? sender, InferenceDiagnostics diagnostics)
        {
            try
            {
                var json = System.Text.Json.JsonSerializer.Serialize(new
                {
                    isGenerating = diagnostics.IsGenerating,
                    status = diagnostics.Status,
                    tokensPerSecond = Math.Round(diagnostics.TokensPerSecond, 2),
                    generatedTokenCount = diagnostics.GeneratedTokenCount,
                    contextUtilization = Math.Round(diagnostics.ContextUtilization, 1),
                    averageEntropy = Math.Round(diagnostics.AverageTokenEntropy, 4),
                    lastToken = diagnostics.RecentTokens.LastOrDefault()?.TokenText ?? ""
                });
                Response.WriteAsync($"data: {json}\n\n", cancellationToken);
                Response.Body.FlushAsync(cancellationToken);
            }
            catch { }
        }

        _diagnosticsService.DiagnosticsUpdated += OnDiagnosticsUpdated;

        try
        {
            // Keep connection open until cancelled
            await Task.Delay(Timeout.Infinite, cancellationToken);
        }
        catch (OperationCanceledException)
        {
            // Expected when client disconnects
        }
        finally
        {
            _diagnosticsService.DiagnosticsUpdated -= OnDiagnosticsUpdated;
        }
    }

    [HttpGet]
    public IActionResult GetDebugLog()
    {
        var entries = _debugLog.GetEntries()
            .Select(e => new
            {
                id = e.Id,
                timestamp = e.Timestamp,
                source = e.Source,
                message = e.Message
            });

        return Json(entries);
    }
}

public class AddDocumentRequest
{
    public string Content { get; set; } = string.Empty;
    public Dictionary<string, string>? Metadata { get; set; }
}
