$folder = "F:\data\spears\tomo_sample_2_051_rec"
$output = "r2_cross_stack.csv"

$first = $true

for ($start = 0; $start -le 2400; $start += 100) {
    $end = $start + 10
    $samples = "${start}:${end}"

    Write-Host "Running samples $samples"

    if ($first) {
        python measure_2_regions.py `
            "$folder" `
			--air 250:350 740:840 `
			--metal 600:1200 1400:1600 `
            --samples $samples `
            --output "$output"

        $first = $false
    }
    else {
        python measure_2_regions.py `
            "$folder" `
			--air 250:350 740:840 `
			--metal 600:1200 1400:1600 `
            --samples $samples `
            --output "$output" `
            --append
    }
}

Write-Host "Done. Results written to $output"