using LocalLLMChat.Models;
using LocalLLMChat.Services;
using Microsoft.AspNetCore.Mvc;

namespace LocalLLMChat.Controllers;

public class SettingsController : Controller
{
    private const int MaxContextLength = 262_144;
    private readonly ILlmSettingsStore _settingsStore;
    private readonly ILlmService _llmService;
    private readonly ILogger<SettingsController> _logger;

    public SettingsController(
        ILlmSettingsStore settingsStore,
        ILlmService llmService,
        ILogger<SettingsController> logger)
    {
        _settingsStore = settingsStore;
        _llmService = llmService;
        _logger = logger;
    }

    [HttpGet]
    public IActionResult Llm()
    {
        var settings = _settingsStore.Get();
        return View(settings);
    }

    [HttpGet]
    public IActionResult GetLlm()
    {
        var settings = _settingsStore.Get();
        return Json(settings);
    }

    [HttpPost]
    public async Task<IActionResult> SaveLlm([FromBody] LlmSettings incoming)
    {
        var current = _settingsStore.Get();
        var merged = Merge(current, incoming ?? new LlmSettings());
        var errors = Validate(merged);

        if (errors.Count > 0)
        {
            return BadRequest(new { ok = false, errors });
        }

        try
        {
            var result = await _llmService.ApplySettingsAsync(merged);
            return Json(new { ok = true, reloaded = result.Reloaded, settings = result.Settings });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to apply LLM settings");
            return StatusCode(500, new { ok = false, errors = new[] { ex.Message } });
        }
    }

    private static LlmSettings Merge(LlmSettings current, LlmSettings incoming)
    {
        var merged = current.Clone();

        if (!string.IsNullOrWhiteSpace(incoming.ModelPath))
        {
            merged.ModelPath = incoming.ModelPath;
        }

        merged.CpuOnly = incoming.CpuOnly;
        merged.ContextLength = incoming.ContextLength;
        merged.GpuOffloadLayers = incoming.GpuOffloadLayers;
        merged.CpuThreadPoolSize = incoming.CpuThreadPoolSize;
        merged.EvaluationBatchSize = incoming.EvaluationBatchSize;
        merged.RopeFrequencyBaseEnabled = incoming.RopeFrequencyBaseEnabled;
        merged.RopeFrequencyBase = incoming.RopeFrequencyBase;
        merged.RopeFrequencyScaleEnabled = incoming.RopeFrequencyScaleEnabled;
        merged.RopeFrequencyScale = incoming.RopeFrequencyScale;
        merged.OffloadKvCacheToGpu = incoming.OffloadKvCacheToGpu;
        merged.KeepModelInMemory = incoming.KeepModelInMemory;
        merged.TryMmap = incoming.TryMmap;
        merged.UseMlock = incoming.UseMlock;
        merged.RandomSeed = incoming.RandomSeed;
        merged.Seed = incoming.Seed;
        merged.FlashAttention = incoming.FlashAttention;
        merged.KCacheQuantTypeEnabled = incoming.KCacheQuantTypeEnabled;
        merged.KCacheQuantType = incoming.KCacheQuantType;
        merged.VCacheQuantTypeEnabled = incoming.VCacheQuantTypeEnabled;
        merged.VCacheQuantType = incoming.VCacheQuantType;
        merged.OverrideDomainType = string.IsNullOrWhiteSpace(incoming.OverrideDomainType)
            ? merged.OverrideDomainType
            : incoming.OverrideDomainType;

        return merged;
    }

    private static List<string> Validate(LlmSettings settings)
    {
        var errors = new List<string>();

        if (settings.ContextLength < 256 || settings.ContextLength > MaxContextLength)
        {
            errors.Add($"Context length must be between 256 and {MaxContextLength} tokens.");
            settings.ContextLength = Math.Clamp(settings.ContextLength, 256, MaxContextLength);
        }

        if (settings.CpuThreadPoolSize < -1 || settings.CpuThreadPoolSize > 128)
        {
            errors.Add("CPU thread pool size must be between -1 (auto) and 128.");
        }

        if (settings.EvaluationBatchSize < 32 || settings.EvaluationBatchSize > 4096)
        {
            errors.Add("Evaluation batch size must be between 32 and 4096.");
            settings.EvaluationBatchSize = Math.Clamp(settings.EvaluationBatchSize, 32, 4096);
        }

        if (settings.GpuOffloadLayers < 0)
        {
            settings.GpuOffloadLayers = 0;
        }
        else if (settings.GpuOffloadLayers > 999)
        {
            settings.GpuOffloadLayers = 999;
        }

        if (settings.RopeFrequencyBaseEnabled)
        {
            if (!settings.RopeFrequencyBase.HasValue || settings.RopeFrequencyBase.Value <= 0 || settings.RopeFrequencyBase.Value > 1_000_000)
            {
                errors.Add("RoPE frequency base must be between 0 and 1,000,000 when enabled.");
            }
        }

        if (settings.RopeFrequencyScaleEnabled)
        {
            if (!settings.RopeFrequencyScale.HasValue || settings.RopeFrequencyScale.Value <= 0 || settings.RopeFrequencyScale.Value > 10_000)
            {
                errors.Add("RoPE frequency scale must be between 0 and 10,000 when enabled.");
            }
        }

        if (!settings.RandomSeed && settings.Seed < 0)
        {
            errors.Add("Seed must be zero or positive when Random Seed is disabled.");
            settings.Seed = Math.Max(0, settings.Seed);
        }

        if (settings.CpuOnly)
        {
            settings.GpuOffloadLayers = 0;
            settings.OffloadKvCacheToGpu = false;
        }

        return errors;
    }
}
