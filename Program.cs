using LocalLLMChat.Models;
using LocalLLMChat.Services;
using LocalLLMChat.Services.Plugins;
using LocalLLMChat.Rag;
using LocalLLMChat.Rag.DocumentLoaders;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddControllersWithViews();

// Configure LLM settings from appsettings.json
builder.Services.Configure<LlmSettings>(
    builder.Configuration.GetSection("LlmSettings"));

// RAG configuration
var ragConfig = new RagConfig();
builder.Configuration.GetSection("Rag").Bind(ragConfig);
builder.Services.AddSingleton(ragConfig);

// Register core services
builder.Services.AddSingleton<IDiagnosticsService, DiagnosticsService>();
builder.Services.AddSingleton<IServerDebugLog, ServerDebugLog>();
builder.Services.AddSingleton<ILlmSettingsStore, JsonLlmSettingsStore>();
builder.Services.AddSingleton<ILlmService, LlamaSharpService>();
builder.Services.AddSingleton<TextChunker>();
builder.Services.AddSingleton<GgufEmbeddingGenerator>();
builder.Services.AddSingleton<SqliteVecStore>();
builder.Services.AddSingleton<PromptBuilder>();
builder.Services.AddSingleton<ITextDocumentLoader, TextFileLoader>();
builder.Services.AddSingleton<ITextDocumentLoader, PdfLoader>();
builder.Services.AddSingleton<RagPipeline>();
builder.Services.AddSingleton<IRagService, RagPlaceholderService>();
builder.Services.AddSingleton<IChatService, ChatService>();

// Register plugins
builder.Services.AddSingleton<IChatPlugin, SystemPromptPlugin>();
builder.Services.AddSingleton<IChatPlugin, RagPlugin>();
builder.Services.AddSingleton<IChatPlugin, ConversationHistoryPlugin>();

var app = builder.Build();

// Configure the HTTP request pipeline
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Chat}/{action=Index}/{id?}");

app.Run();
