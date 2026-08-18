$folder = "F:\data\spears\tomo_sample_2_051_rec"
$output = "im_cross_stack.csv"
$downsample = 100

$first = $true

for ($start = 0; $start -le 2400; $start += 100) {
    $end = $start + 10
    $samples = "${start}:${end}"

    Write-Host "Running samples $samples"

    if ($first) {
        python measure_im.py `
            "$folder" `
            $downsample `
			--smooth 2 `
			--peak-window 64 `
			--min-peak-distance 128 `
            --samples $samples `
            --output "$output"

        $first = $false
    }
    else {
        python measure_im.py `
            "$folder" `
            $downsample `
			--smooth 2 `
			--peak-window 64 `
			--min-peak-distance 128 `
            --samples $samples `
            --output "$output" `
            --append
    }
}

Write-Host "Done. Results written to $output"