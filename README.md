# SubtitleFlow — STT ve Türkçe MT

Youtube linki girilir, çeviri başlatılır sonrasında eğer kullanıcı tarafından istenmiş ise altyazılı video oluşturulur.

## macOS: bir kere yapılacak kurulum

Komutları **Terminal**'de çalıştırın; depo kökü gerektirenler belirtilmiştir.

**1. Homebrew (yoksa).** Resmî sayfa: <https://brew.sh/>. Resmî kurulum komutu:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Komut satırı araçları eksikse önce `xcode-select --install` çalıştırabilirsiniz.

**2. Araçları kurun.**

```bash
brew install python@3.12 uv yt-dlp deno
brew install ffmpeg-full
```

Sıradan `ffmpeg` formülü libass içermez; **kalıcı altyazı (burn-in) için
`ffmpeg-full` gerekir.** `ffmpeg-full` keg-only olduğundan `PATH`'e elle
eklenmelidir:

```bash
echo 'export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**3. Proje bağımlılıklarını eşitleyin.** Depo kökünde:

```bash
uv sync --locked --extra api --extra cli --extra desktop
```

**4. Kurulumu doğrulayın.** Her komut sürüm yazmalı; `subtitles` satırı **mutlaka
dönmelidir**:

```bash
python3.12 --version
uv --version
ffmpeg -version
ffprobe -version
ffmpeg -filters | grep subtitles
yt-dlp --version
deno --version
```

`deno`, güncel `yt-dlp`'nin YouTube JavaScript challenge çözümü için gereklidir;
resmî belgeler onu önerilen ve varsayılan etkin çalışma zamanı sayar.

## Windows: bir kere yapılacak kurulum

Hedef yol **Windows 10/11 x64**'tür. **Windows üzerinde fiziksel QA hâlâ
açıktır**; bu bölüm kurulum talimatıdır, doğrulanmış çalıştırma kanıtı değildir.
Komutları **PowerShell**'de çalıştırın.

**1. Araçları kurun.** `-e` tam paket kimliğini eşler; kabul bayrakları soruları
otomatik onaylar:

```powershell
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
winget install -e --id astral-sh.uv --accept-source-agreements --accept-package-agreements
winget install -e --id Gyan.FFmpeg.Essentials --accept-source-agreements --accept-package-agreements
winget install -e --id yt-dlp.yt-dlp --accept-source-agreements --accept-package-agreements
winget install -e --id DenoLand.Deno --accept-source-agreements --accept-package-agreements
```

**2. WebView2 Runtime (yalnız gerekiyorsa).** Masaüstü penceresi
`Microsoft.WebView2` kullanır; Windows 11 ve çoğu Windows 10 sisteminde hazırdır
ama garanti değildir. Eksikse
<https://developer.microsoft.com/microsoft-edge/webview2/> adresinden veya şu
komutla kurun:

```powershell
winget install -e --id Microsoft.EdgeWebView2Runtime --accept-source-agreements --accept-package-agreements
```

**3. PowerShell'i kapatıp yeniden açın** ki `PATH` güncellensin.

**4. Kurulumu doğrulayın.** Gyan Essentials libass içerir; yine de teyit edin ve
`subtitles` satırının döndüğünü görün:

```powershell
python --version
uv --version
ffmpeg -version
ffprobe -version
ffmpeg -filters | Select-String subtitles
yt-dlp --version
deno --version
```

`python --version` 3.12 göstermiyorsa `py -3.12 --version` kullanın.

**5. Proje bağımlılıklarını eşitleyin.** Depo kökünde:

```powershell
uv sync --locked --extra api --extra cli --extra desktop
```

## API hesapları, anahtarlar ve .env

İki hesap gerekir: konuşma-metni için ElevenLabs, çeviri için Google Cloud.
Google Cloud projesinde **faturalandırmanın (billing) etkin olması gerekir**;
hizmet koşulları bunu şart koşar. ElevenLabs API'sinin kullanılabilirliği ve
kullanımı hesabınızın güncel planına/kotasına ya da kullandıkça öde düzeninize
bağlıdır; çalıştırmadan önce hesabınızı doğrulayın. Ücretsiz kullanım sözü
verilmez.

### ElevenLabs API anahtarı


1. <https://elevenlabs.io/app/developers/api-keys> adresinden hesap oluşturun veya giriş yapın.
2. Sol menüden **Developers → API Keys**'e gidin.
3. **Create API key** ile kısıtlı bir anahtar üretin; **Speech-to-Text/Scribe**
   erişimini etkinleştirin. İsteğe bağlı bir kredi limiti koyabilirsiniz.
4. Tam anahtar **yalnızca bir kez** gösterilir; hemen `ELEVENLABS_API_KEY`
   değerine kopyalayın.

### Google Cloud Translation API anahtarı


1. <https://console.cloud.google.com/> üzerinde bir proje oluşturun veya seçin.
2. Projede **faturalandırmayı etkinleştirin** (billing hesabı bağlayın).
3. **Cloud Translation API**'yi etkinleştirin: API kimliği
   `translate.googleapis.com`.
4. **APIs & Services → Credentials → Create credentials → API key** ile anahtar
   üretin.
5. Anahtarı **Cloud Translation API** ile sınırlayın; tercihen bir uygulama
   ve/veya IP kısıtı da ekleyin.
6. `GOOGLE_TRANSLATION_PROJECT` değeri projenin **Project ID**'sidir; görünen
   ad (display name) veya proje numarası değildir.

### `.env` dosyası

`.env.example` dosyasını kopyalayıp doldurun.

macOS:

```bash
cp .env.example .env
open -e .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Gerekli minimum içerik (köşeli parantezler yer tutucudur, gerçek değer değildir):

```dotenv
ELEVENLABS_API_KEY=<elevenlabs-api-key>
GOOGLE_TRANSLATION_PROJECT=<google-cloud-project-id>
GOOGLE_TRANSLATION_API_KEY=<google-translation-api-key>
```

- `=` işaretinin çevresine tırnak veya boşluk koymayın.
- `.env` Git tarafından yok sayılır; **asla commit edilmemeli veya
  paylaşılmamalıdır.**

### ElevenLabs Scribe v2

Resmî API fiyatı (batch): **`$0.22 / ses saati`**. Faturalandırma USD'dir ve
fiyatlara vergi dahil değildir. Keyterm (özel terim) kullanılırsa ek
**`$0.05 / ses saati`**. Kaynak: <https://elevenlabs.io/pricing/api>.

Süre bazlı hesap (keyterm'siz):

- 10 dakika = 600 sn → `600 / 3600 × $0.22 = $0.0367`
- 1 saat = 3600 sn → `$0.22`

### Google Translation LLM

Resmî fiyat: **`$10 / 1.000.000 girdi karakteri`** + **`$10 / 1.000.000 çıktı
karakteri`**. Her karakter bir kod noktasıdır; boşluklar da sayılır. Kaynak:
<https://cloud.google.com/translate/pricing> ve model açıklaması
<https://cloud.google.com/translate/docs/translation-llm>.

- 50.000 girdi + 50.000 çıktı = 100.000 → `$1.00` gibi

## macOS'ta çalıştırma

Depo kökünden: Finder'da **`Transkript.command`** dosyasına çift tıklayın veya
Terminalden:

```bash
./Transkript.command
```

Açılmazsa `chmod +x Transkript.command` ile çalıştırma izni verin. Başlatıcı
çalışmazsa açık komut:

```bash
uv run --extra api --extra cli --extra desktop subtitle-flow desktop
```

## Windows'ta çalıştırma

Depo kökünde **`Transkript.cmd`** dosyasına çift tıklayın veya PowerShell'den:

```powershell
./Transkript.cmd
```

Başlatıcı çalışmazsa açık komut:

```powershell
uv run --extra api --extra cli --extra desktop subtitle-flow desktop
```

## Arayüz nasıl kullanılır

Tek bir **herkese açık YouTube bağlantısı** yapıştırın;  **ücretli çağrı onayını**
işaretleyip **bütçe/rezervasyon** değerlerini girin; isterseniz **"Türkçe
altyazılı video oluştur"** kutusunu işaretleyin (varsayılan **kapalıdır**) ve
**Transkripti al ve Türkçeye çevir**'e basın. Çıktı yolları (depo köküne göre):

- Kaynak Markdown: `outputs/transkriptler/<video_id>.md`
- Türkçe Markdown: `outputs/ceviriler/<video_id>.md`
- Altyazılı video (burn seçildiyse):
  `outputs/videolar/<video_id>-turkce-altyazili.mp4`
