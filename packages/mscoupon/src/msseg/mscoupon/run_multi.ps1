$folder = "F:\data\spears\tomo_sample_2_051_rec"
$output = "gmm_cross_stack.csv"
$downsample = 100

$first = $true

for ($start = 0; $start -le 2400; $start += 100) {
    $end = $start + 10
    $samples = "${start}:${end}"

    Write-Host "Running samples $samples"

    if ($first) {
        python measure_gmm.py `
            "$folder" `
            $downsample `
			--trim 0.5 `
            --samples $samples `
            --output "$output"

        $first = $false
    }
    else {
        python measure_gmm.py `
            "$folder" `
            $downsample `
			--trim 0.5 `
            --samples $samples `
            --output "$output" `
            --append
    }
}

Write-Host "Done. Results written to $output"