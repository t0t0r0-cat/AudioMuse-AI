<#
Run the Jellyfin diagnostic script with proper environment variables.

Usage examples (PowerShell):
  # Provide values as parameters
  .\tools\run_jellyfin_debug.ps1 -JellyfinUrl 'http://your_jellyfin:8096' -JellyfinToken 'YOUR_TOKEN' -JellyfinUserId 'YOUR_USER_ID'

  # Prompt for values interactively
  .\tools\run_jellyfin_debug.ps1

  # Run diagnostics for both Recursive True and False (helps compare behavior)
  .\tools\run_jellyfin_debug.ps1 -JellyfinUrl 'http://your_jellyfin:8096' -JellyfinToken 'YOUR_TOKEN' -JellyfinUserId 'YOUR_USER_ID' -TestBoth

Notes:
- This script only sets environment variables for the current PowerShell process and runs the Python diagnostic.
- Ensure you run it from the repository root where the project and Python environment are available.
#>

param(
    [string]$JellyfinUrl,
    [string]$JellyfinToken,
    [string]$JellyfinUserId,
    [switch]$TestBoth
)

function PromptIfEmpty([string]$value, [string]$prompt) {
    if ([string]::IsNullOrWhiteSpace($value)) {
        return Read-Host -Prompt $prompt
    }
    return $value
}

try {
    $JellyfinUrl = PromptIfEmpty $JellyfinUrl "Jellyfin URL (e.g. http://your_jellyfin:8096)"
    $JellyfinToken = PromptIfEmpty $JellyfinToken "Jellyfin Token"
    $JellyfinUserId = PromptIfEmpty $JellyfinUserId "Jellyfin User ID"

    # Basic validation to avoid running against the placeholder/default URL
    function IsPlaceholderUrl($u) {
        if (-not $u) { return $true }
        $lower = $u.ToLower()
        return $lower.Contains('your_jellyfin') -or $lower.Contains('your_jellyfin_url') -or $lower.Contains('your_jellyfin:8096') -or $lower.Contains('http://your_jellyfin')
    }

    if (IsPlaceholderUrl $JellyfinUrl) {
        Write-Host "The Jellyfin URL you provided looks like the placeholder/default value: $JellyfinUrl" -ForegroundColor Yellow
        $confirm = Read-Host -Prompt "Do you want to continue anyway? Type 'yes' to continue or 'no' to re-enter the URL"
        if ($confirm.Trim().ToLower() -ne 'yes') {
            $JellyfinUrl = Read-Host -Prompt "Enter the real Jellyfin URL (e.g. http://192.168.1.100:8096)"
            if (IsPlaceholderUrl $JellyfinUrl) {
                Write-Host "Still looks like a placeholder. Aborting to avoid running diagnostics against the wrong server." -ForegroundColor Red
                exit 1
            }
        }
    }

    if ($TestBoth) {
        foreach ($rec in @('True','False')) {
            Write-Host "\n=== Running jellyfin_debug.py with JELLYFIN_RECURSIVE=$rec ===\n" -ForegroundColor Cyan
            $env:JELLYFIN_URL = $JellyfinUrl
            $env:JELLYFIN_TOKEN = $JellyfinToken
            $env:JELLYFIN_USER_ID = $JellyfinUserId
            $env:JELLYFIN_RECURSIVE = $rec

            python .\tools\jellyfin_debug.py
            if ($LASTEXITCODE -ne 0) {
                Write-Host "jellyfin_debug.py exited with code $LASTEXITCODE" -ForegroundColor Yellow
            }
        }
    }
    else {
        $env:JELLYFIN_URL = $JellyfinUrl
        $env:JELLYFIN_TOKEN = $JellyfinToken
        $env:JELLYFIN_USER_ID = $JellyfinUserId
        # Keep JELLYFIN_RECURSIVE as whatever is set in your env or default in config

        python .\tools\jellyfin_debug.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "jellyfin_debug.py exited with code $LASTEXITCODE" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "Error running the diagnostic: $_" -ForegroundColor Red
    exit 1
}
