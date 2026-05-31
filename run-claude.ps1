param(
    [string]$ClaudeDir = ".",
    [switch]$NoStart
)

$GeminiUrl = "http://localhost:8081"
$AdapterUrl = "http://localhost:8082"
$ApiKey = "sk-gemini"
$BaseDir = "C:\Users\pasaz\PycharmProjects\GeminiClaudeCode"

if (-not $NoStart) {
    $geminiRunning = Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*gemini_web2api*" }
    $adapterRunning = Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*anthropic2openai*" }

    if (-not $geminiRunning) {
        Write-Host "[*] Starting gemini-web2api..." -ForegroundColor Cyan
        Start-Job -Name GeminiWeb2Api -ScriptBlock {
            param($dir)
            Set-Location "$dir\gemini-web2api"
            python gemini_web2api.py
        } -ArgumentList $BaseDir | Out-Null
        Start-Sleep -Seconds 4
        Write-Host "[+] gemini-web2api at $GeminiUrl" -ForegroundColor Green
    } else {
        Write-Host "[+] gemini-web2api already running at $GeminiUrl" -ForegroundColor Green
    }

    if (-not $adapterRunning) {
        Write-Host "[*] Starting anthropic→openai adapter..." -ForegroundColor Cyan
        Start-Job -Name Anthropic2OpenAI -ScriptBlock {
            param($dir)
            Set-Location $dir
            python anthropic2openai.py
        } -ArgumentList $BaseDir | Out-Null
        Start-Sleep -Seconds 2
        Write-Host "[+] Adapter at $AdapterUrl" -ForegroundColor Green
    } else {
        Write-Host "[+] Adapter already running at $AdapterUrl" -ForegroundColor Green
    }
}

$env:ANTHROPIC_BASE_URL = $AdapterUrl
$env:ANTHROPIC_API_KEY = $ApiKey
$env:OPENAI_BASE_URL = "$GeminiUrl/v1"
$env:OPENAI_API_KEY = $ApiKey
$env:CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = "gemini-3.5-flash-thinking"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "gemini-3.5-flash"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "gemini-flash-lite"
$env:CLAUDE_CODE_SUBAGENT_MODEL = "gemini-3.5-flash"

Write-Host "[*] Environment configured:" -ForegroundColor Cyan
Write-Host "    ANTHROPIC_BASE_URL  = $AdapterUrl" -ForegroundColor Gray
Write-Host "    ANTHROPIC_API_KEY   = sk-****" -ForegroundColor Gray
Write-Host "    OPENAI_BASE_URL     = $GeminiUrl/v1" -ForegroundColor Gray
Write-Host "    DEFAULT_OPUS        = gemini-3.5-flash-thinking" -ForegroundColor Gray
Write-Host "    DEFAULT_SONNET      = gemini-3.5-flash" -ForegroundColor Gray
Write-Host "    DEFAULT_HAIKU       = gemini-flash-lite" -ForegroundColor Gray
Write-Host ""
Write-Host "[*] Opening Claude Code in $ClaudeDir ..." -ForegroundColor Cyan

Set-Location $ClaudeDir
claude
