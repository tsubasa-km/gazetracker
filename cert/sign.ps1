param(
    [string]$Exe = "$PSScriptRoot\..\dist\GazeTracker\GazeTracker.exe",
    [string]$Subject = "CN=GazeTracker Self-Signed"
)
# 自己署名のコード署名証明書で exe に署名する。証明書が無ければ作る。
$ErrorActionPreference = "Stop"

$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq $Subject } |
    Sort-Object NotAfter -Descending | Select-Object -First 1
if (-not $cert) {
    Write-Host "証明書が無いので作成します: $Subject"
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $Subject -KeyAlgorithm RSA -KeyLength 3072 `
        -HashAlgorithm SHA256 -KeyUsage DigitalSignature -CertStoreLocation Cert:\CurrentUser\My `
        -NotAfter (Get-Date).AddYears(5) -FriendlyName "GazeTracker code signing"
}
$cerPath = Join-Path $PSScriptRoot "GazeTracker.cer"
Export-Certificate -Cert $cert -FilePath $cerPath -Type CERT | Out-Null
Write-Host "公開証明書: $cerPath (拇印 $($cert.Thumbprint), 有効期限 $($cert.NotAfter.ToString('yyyy-MM-dd')))"

if (-not (Test-Path $Exe)) { throw "exe が見つかりません: $Exe" }
$r = $null
foreach ($ts in @("http://timestamp.digicert.com", "http://timestamp.sectigo.com", $null)) {
    if ($ts) {
        $r = Set-AuthenticodeSignature -FilePath $Exe -Certificate $cert -HashAlgorithm SHA256 -TimestampServer $ts
    } else {
        Write-Warning "タイムスタンプサーバーに届かないので、タイムスタンプ無しで署名します (証明書の期限切れ後に署名が無効になります)"
        $r = Set-AuthenticodeSignature -FilePath $Exe -Certificate $cert -HashAlgorithm SHA256
    }
    # 自己署名なので、この PC がルートを信頼していなければ Status は UnknownError になる。
    # 署名者が自分の証明書なら署名は付いているので成功とみなす。
    $signed = $r.SignerCertificate -and ($r.SignerCertificate.Thumbprint -eq $cert.Thumbprint)
    if ($signed) { break }
}
if (-not $signed) { throw "署名に失敗: $($r.Status) $($r.StatusMessage)" }
$v = Get-AuthenticodeSignature $Exe
Write-Host "署名しました: $Exe"
Write-Host "  署名者: $($v.SignerCertificate.Subject)"
if ($v.TimeStamperCertificate) { Write-Host "  タイムスタンプ: $($v.TimeStamperCertificate.Subject)" }
if ($v.Status -eq "Valid") {
    Write-Host "  検証: 有効 (この PC は証明書を信頼済み)"
} else {
    Write-Host "  検証: この PC ではまだ未信頼 ($($v.Status))。cert\install_cert.bat を管理者として実行すると有効になります"
}
