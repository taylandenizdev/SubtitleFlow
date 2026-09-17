#!/bin/sh
# Transkript: dosyaya çift tıklayınca arayüzü kendi masaüstü penceresinde açar.
#
# 'uv run' proje bağımlılıklarını seçilen eklerle eşitler (senkronlar); bu betik
# kendiliğinden başka bir araç (ffmpeg, yt-dlp, WebView2 Runtime vb.) kurmaz.
# Sağlayıcı anahtarları arayüze/loglara yazılmaz.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "Hata: 'uv' bulunamadı; önce uv kurulmalıdır." \
    "Kurulum için: https://docs.astral.sh/uv/ ardından bu dosyayı yeniden açın." >&2
  if [ -t 0 ]; then
    printf '%s\n' "Kapatmak için bir tuşa basın..." >&2
    read -r _ || true
  fi
  exit 1
fi

# Varsayılan giriş masaüstü penceresidir; tarayıcı arayüzü için:
#   subtitle-flow ui
exec uv run --extra api --extra cli --extra desktop subtitle-flow desktop "$@"
