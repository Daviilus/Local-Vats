param(
    [string]$OutputDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ssl'),
    [switch]$SkipTrust
)

$ErrorActionPreference = 'Stop'
$output = [IO.Path]::GetFullPath($OutputDirectory)
$rootPath = Join-Path $output 'lvats-local-root.cer'
$certPath = Join-Path $output 'lvats-cert.pem'
$keyPath = Join-Path $output 'lvats-key.pem'

function Get-DerLength([int]$Length) {
    if ($Length -lt 128) { return [byte[]]@([byte]$Length) }
    $bytes = [BitConverter]::GetBytes($Length)
    [Array]::Reverse($bytes)
    $first = 0
    while (($first -lt ($bytes.Length - 1)) -and ($bytes[$first] -eq 0)) { $first++ }
    $value = [byte[]]$bytes[$first..($bytes.Length - 1)]
    return [byte[]](@([byte](0x80 -bor $value.Length)) + $value)
}

function ConvertTo-DerInteger([byte[]]$Value) {
    $first = 0
    while (($first -lt ($Value.Length - 1)) -and ($Value[$first] -eq 0)) { $first++ }
    $valueBytes = [byte[]]$Value[$first..($Value.Length - 1)]
    if (($valueBytes[0] -band 0x80) -ne 0) {
        $valueBytes = [byte[]](@(0) + $valueBytes)
    }
    return [byte[]](@(0x02) + (Get-DerLength $valueBytes.Length) + $valueBytes)
}

function Export-RsaPrivateKeyPkcs1([Security.Cryptography.RSA]$Key) {
    $parameters = $Key.ExportParameters($true)
    $body = [Collections.Generic.List[byte]]::new()
    $body.AddRange([byte[]]@(0x02, 0x01, 0x00))
    foreach ($part in @(
        $parameters.Modulus, $parameters.Exponent, $parameters.D, $parameters.P,
        $parameters.Q, $parameters.DP, $parameters.DQ, $parameters.InverseQ
    )) {
        $body.AddRange([byte[]](ConvertTo-DerInteger $part))
    }
    $result = [Collections.Generic.List[byte]]::new()
    $result.Add(0x30)
    $result.AddRange([byte[]](Get-DerLength $body.Count))
    $result.AddRange($body.ToArray())
    return $result.ToArray()
}

function ConvertTo-Pem([string]$Label, [byte[]]$Bytes) {
    $base64 = [Convert]::ToBase64String($Bytes, [Base64FormattingOptions]::InsertLineBreaks)
    return "-----BEGIN $Label-----`r`n$base64`r`n-----END $Label-----`r`n"
}

function Install-LvatsRoot([string]$Path) {
    if ($SkipTrust) { return }
    $certificate = [Security.Cryptography.X509Certificates.X509Certificate2]::new($Path)
    $store = [Security.Cryptography.X509Certificates.X509Store]::new(
        [Security.Cryptography.X509Certificates.StoreName]::Root,
        [Security.Cryptography.X509Certificates.StoreLocation]::CurrentUser
    )
    try {
        $store.Open([Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
        $found = $store.Certificates | Where-Object Thumbprint -eq $certificate.Thumbprint
        if (-not $found) { $store.Add($certificate) }
    } finally {
        $store.Close()
        $certificate.Dispose()
    }
}

if ((Test-Path -LiteralPath $rootPath) -and
    (Test-Path -LiteralPath $certPath) -and
    (Test-Path -LiteralPath $keyPath)) {
    Install-LvatsRoot $rootPath
    Write-Output "Lvats HTTPS certificate is ready: $certPath"
    exit 0
}

[IO.Directory]::CreateDirectory($output) | Out-Null
$notBefore = [DateTimeOffset]::UtcNow.AddMinutes(-5)
$rootKey = [Security.Cryptography.RSA]::Create(3072)
$leafKey = [Security.Cryptography.RSA]::Create(2048)
$root = $null
$leaf = $null
$leafWithKey = $null
try {
    $rootRequest = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
        'CN=Lvats Local Root CA', $rootKey,
        [Security.Cryptography.HashAlgorithmName]::SHA256,
        [Security.Cryptography.RSASignaturePadding]::Pkcs1
    )
    $rootRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($true, $false, 0, $true)
    )
    $rootRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new(
            [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign -bor
            [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::CrlSign,
            $true
        )
    )
    $rootRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509SubjectKeyIdentifierExtension]::new($rootRequest.PublicKey, $false)
    )
    $root = $rootRequest.CreateSelfSigned($notBefore, $notBefore.AddYears(10))

    $leafRequest = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
        'CN=localhost', $leafKey,
        [Security.Cryptography.HashAlgorithmName]::SHA256,
        [Security.Cryptography.RSASignaturePadding]::Pkcs1
    )
    $leafRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($false, $false, 0, $true)
    )
    $leafRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new(
            [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature -bor
            [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyEncipherment,
            $true
        )
    )
    $oids = [Security.Cryptography.OidCollection]::new()
    [void]$oids.Add([Security.Cryptography.Oid]::new('1.3.6.1.5.5.7.3.1'))
    $leafRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new($oids, $true)
    )
    $san = [Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
    $san.AddDnsName('localhost')
    $san.AddIpAddress([Net.IPAddress]::Parse('127.0.0.1'))
    $san.AddIpAddress([Net.IPAddress]::Parse('::1'))
    $leafRequest.CertificateExtensions.Add($san.Build())
    $leafRequest.CertificateExtensions.Add(
        [Security.Cryptography.X509Certificates.X509SubjectKeyIdentifierExtension]::new($leafRequest.PublicKey, $false)
    )

    $serial = [byte[]]::new(16)
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $random.GetBytes($serial) } finally { $random.Dispose() }
    $leaf = $leafRequest.Create($root, $notBefore, $notBefore.AddYears(3), $serial)
    $leafWithKey = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::CopyWithPrivateKey($leaf, $leafKey)

    [IO.File]::WriteAllBytes($rootPath, $root.Export([Security.Cryptography.X509Certificates.X509ContentType]::Cert))
    [IO.File]::WriteAllText($certPath, (ConvertTo-Pem 'CERTIFICATE' $leafWithKey.RawData), [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($keyPath, (ConvertTo-Pem 'RSA PRIVATE KEY' (Export-RsaPrivateKeyPkcs1 $leafKey)), [Text.UTF8Encoding]::new($false))
} finally {
    if ($leafWithKey) { $leafWithKey.Dispose() }
    if ($leaf) { $leaf.Dispose() }
    if ($root) { $root.Dispose() }
    $leafKey.Dispose()
    $rootKey.Dispose()
}

Install-LvatsRoot $rootPath
Write-Output "Lvats HTTPS certificate created for localhost and 127.0.0.1: $certPath"
