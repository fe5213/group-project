# Rebuild all numerical outputs and (if available) the PDF report.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> python main.py --use-frozen" -ForegroundColor Cyan
python main.py --use-frozen

if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host "==> LaTeX build" -ForegroundColor Cyan
    Set-Location "$root\report"
    pdflatex -interaction=nonstopmode fe5213_report.tex
    if (Get-Command bibtex -ErrorAction SilentlyContinue) {
        bibtex fe5213_report
    }
    pdflatex -interaction=nonstopmode fe5213_report.tex
    pdflatex -interaction=nonstopmode fe5213_report.tex
    Set-Location $root
    Write-Host "Done. PDF: report\fe5213_report.pdf" -ForegroundColor Green
}
else {
    Write-Host "pdflatex not found; skipped PDF build. Run LaTeX manually from report\" -ForegroundColor Yellow
}
