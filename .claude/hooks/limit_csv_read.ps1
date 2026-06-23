$raw = [Console]::In.ReadToEnd()
$payload = $raw | ConvertFrom-Json
$filePath = $payload.tool_input.file_path

if ($filePath -match '\.csv$') {
    $limit = $payload.tool_input.limit
    if (-not $limit -or [int]$limit -gt 5) {
        Write-Output "Lecture de CSV bloquee : utilisez le parametre 'limit' avec une valeur <= 5 pour ne lire que le debut du fichier (ex: limit=5)."
        exit 2
    }
}
exit 0
