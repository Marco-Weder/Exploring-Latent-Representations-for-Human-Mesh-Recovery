#!/bin/bash
set -euo pipefail

DEST="/cluster/scratch/$USER/thesis_HMR/tokenizer_data"
mkdir -p "$DEST"

urle () {
    [[ "${1:-}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

download_and_unzip() {
    local url="$1"
    local output_file
    output_file="$(basename "$url" | sed 's/.*sfile=//')"
    local zip_path="$DEST/$output_file"

    wget --continue --post-data "username=$username&password=$password" "$url" -O "$zip_path"

    if file "$zip_path" | grep -qi html; then
        echo "Download failed: got HTML instead of ZIP for $zip_path"
        exit 1
    fi

    unzip -o "$zip_path" -d "$DEST"
    rm -f "$zip_path"
}

echo
echo "You need to register at https://tokenhmr.is.tue.mpg.de"
read -r -p "Username: " username_raw
read -r -s -p "Password: " password_raw
echo

username="$(urle "$username_raw")"
password="$(urle "$password_raw")"

download_and_unzip "https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=data.zip"
#download_and_unzip "https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=tokenhmr_model_latest.zip"
download_and_unzip "https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=tokenization_data.zip"

echo "Done. Files extracted under: $DEST"