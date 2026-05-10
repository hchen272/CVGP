# batch_upscale_folders.ps1
# Batch upscale image sequence folders using realesrgan-ncnn-vulkan

$inputDir = "D:\VideoSuperResolution\BasicVSR_PlusPlus\data\reds\input_videos"
$outputRoot = "D:\VideoSuperResolution\ESRGAN\results"
$realesrgan = ".\realesrgan-ncnn-vulkan.exe"

$scale = 4
$model = "realesr-animevideov3"
$frameFormat = "png"

if (!(Test-Path $realesrgan)) {
    Write-Host "ERROR: realesrgan-ncnn-vulkan.exe not found at $realesrgan" -ForegroundColor Red
    exit 1
}

$subFolders = Get-ChildItem -Path $inputDir -Directory
if ($subFolders.Count -eq 0) {
    Write-Host "No subfolders found in $inputDir. Exiting." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($subFolders.Count) subfolders to process." -ForegroundColor Cyan

foreach ($folder in $subFolders) {
    $videoName = $folder.Name
    $inputFolder = $folder.FullName
    # Build output path correctly: outputs/<videoName>/realesrgan_x4
    $outputFolder = Join-Path $outputRoot $videoName
    $outputFolder = Join-Path $outputFolder "realesrgan_x${scale}"
    
    Write-Host "Processing: $videoName" -ForegroundColor Cyan
    Write-Host "  Input:  $inputFolder"
    Write-Host "  Output: $outputFolder"
    
    # Create output directory if not exists
    if (!(Test-Path $outputFolder)) {
        New-Item -ItemType Directory -Force -Path $outputFolder | Out-Null
    }
    
    Write-Host "  Running realesrgan (scale=$scale, model=$model) ..."
    & $realesrgan -i $inputFolder -o $outputFolder -s $scale -n $model -f $frameFormat -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Upscaling failed for $videoName" -ForegroundColor Red
    } else {
        Write-Host "  Done." -ForegroundColor Green
    }
}

Write-Host "All folders processed." -ForegroundColor Green