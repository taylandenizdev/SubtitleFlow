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

İki hesap gerekir: konuşma-metni (STT) için **ElevenLabs**, çeviri için **Google
Cloud**. İki hizmet de ücretlidir; kullanım, hesabınızın güncel planına/kotasına
ya da kullandıkça öde düzeninize bağlıdır. Çalıştırmadan önce hesabınızın
kotasını ve faturalandırmasını doğrulayın. Ücretsiz kullanım sözü verilmez.

Google Cloud projesinde **faturalandırmanın (billing) etkin olması gerekir**;
Cloud Translation API hizmet koşulları bunu şart koşar.

> Not: Google Cloud ve ElevenLabs arayüzlerinin dili ve menü yerleşimi zamanla
> veya hesap dil ayarına göre biraz değişebilir. Aşağıdaki adımlar İngilizce
> arayüzü anlatır; birebir aynı görünmese de aynı işlevi arayın.

Bu masaüstü uygulaması Google isteklerini **sizin makinenizden** yapar. Bu yüzden
aşağıdaki anahtarlar ve uygulama kısıtları tarayıcı değil, bu bilgisayar için
düşünülmelidir.

### ElevenLabs API anahtarı

1. Tarayıcıdan doğrudan anahtar sayfasını açın:
   <https://elevenlabs.io/app/developers/api-keys>. Hesabınız yoksa **Sign up**
   ile oluşturun, varsa **Log in** ile giriş yapın.
2. Sayfa doğrudan açılmazsa sol menüden **Developers** → **API Keys** yolunu izleyin.
3. **Create API key** düğmesine basın.
4. Açılan pencerede anahtara anlamlı bir **isim** verin (örn. `subtitleflow-desktop`).
5. İzinler bölümünde arayüz izin veriyorsa yalnızca **Speech-to-Text** /
   **Scribe** kapsamını seçin; hesabın tamamına yetki veren geniş izinlerden
   kaçının.
6. İsterseniz isteğe bağlı bir **kredi limiti (credit limit)** belirleyin.
7. **Create API key** ile onaylayın. Tam anahtar **yalnızca bir kez** gösterilir;
   pencereyi kapatmadan kopyalayın.
8. Kopyaladığınız değeri `.env` dosyasındaki `ELEVENLABS_API_KEY` satırına
   yapıştırın (aşağıdaki `.env` bölümüne bakın).
9. ElevenLabs panelinde hesabınızın **kotasını/kullanımını** ve gerekiyorsa
   **faturalandırma** durumunu kontrol edin; planınızın bu kullanımı
   karşıladığından emin olun.

### Google Cloud Translation API anahtarı

1. <https://console.cloud.google.com/> adresini açın ve Google hesabınızla giriş
   yapın.
2. Üst çubuktaki **proje seçici**ye (mevcut proje adının göründüğü açılır menü)
   tıklayın.
3. **New Project**'i seçin.
4. **Project name** alanına anlamlı bir ad yazın (örn. `subtitleflow`); kuruluş
   hesabı kullanıyorsanız uygun **Location**'ı seçin ve **Create**'e basın.
5. Projenin oluşmasını bekleyin, ardından üstteki **proje seçici**den bu yeni
   projeyi seçin.
6. **Navigation menu** (sol üstteki üç çizgi) → **Billing** yolunu açın.
   Faturalandırma hesabı bağlı değilse **Link a billing account** deyin; hesap
   yoksa yeni bir faturalandırma hesabı oluşturun ve gerekirse **ödeme profili
   (payment profile)** bilgilerini ekleyin. Projenin faturalandırmaya bağlı
   olduğunu doğrulayın.
7. **Navigation menu** → **APIs & Services** → **Library** yolunu açın.
8. Arama kutusuna **Cloud Translation API** yazın, sonuçtan seçin ve **Enable**
   düğmesine basın. Bu API'nin kimliği `translate.googleapis.com`'dur.
9. **APIs & Services** → **Credentials** sayfasına gidin.
10. Üstteki **Create credentials** → **API key**'i seçin. Anahtar hemen
    oluşturulur ve ekranda gösterilir; kopyalayın.
11. **Hemen** anahtarı kısıtlayın: gösterilen pencerede veya listede anahtarın
    yanındaki **Edit** (kalem) simgesine basın.
12. **API restrictions** altında **Restrict key**'i seçin, listeden **Cloud
    Translation API**'yi işaretleyin ve **Save**'e basın.
13. **Application restrictions** bölümünü dikkatli ayarlayın:
    - **IP addresses**: yalnızca **sabit bir dış IP adresiniz (stable outbound
      IP)** varsa kullanışlıdır; değişken ev/ofis IP'sinde uygulamayı kilitler.
    - **Websites/HTTP referrers** veya **Android/iOS apps**: bu yerel masaüstü
      uygulaması için uygun değildir; tarayıcıdan gelen istekleri varsayar.
      Bunları seçmeyin.
    - Emin değilseniz yalnızca **API restrictions** (Cloud Translation API) ile
      bırakın.
14. `GOOGLE_TRANSLATION_PROJECT` değeri projenin **Project ID**'sidir; görünen ad
    (display name) veya proje numarası değildir. Proje seçicideki projeyi açıp
    **Project ID**'yi tam olarak kopyalayın.
15. İsteğe bağlı: **Billing** → **Budgets & alerts** ile bir bütçe uyarısı kurun.
    Bunun **yalnızca uyarı** verdiğini, harcamayı **otomatik kesmediğini**
    unutmayın; sert bir üst sınır değildir.

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

Dosyaya şu üç satırı yazın (köşeli parantezler yer tutucudur, gerçek değer
değildir):

```dotenv
ELEVENLABS_API_KEY=<elevenlabs-api-key>
GOOGLE_TRANSLATION_PROJECT=<google-cloud-project-id>
GOOGLE_TRANSLATION_API_KEY=<google-translation-api-key>
```

Şimdi **köşeli parantezleri (`<` ve `>` dahil) silin** ve her değeri `=` işaretinin
hemen ardına yapıştırın. Her değerin nereden geldiği:

- `ELEVENLABS_API_KEY` → ElevenLabs **API Keys** sayfasında oluşturduğunuz anahtar.
- `GOOGLE_TRANSLATION_PROJECT` → Google Cloud projesinin **Project ID**'si.
- `GOOGLE_TRANSLATION_API_KEY` → Google Cloud **Credentials** sayfasında oluşturup
  kısıtladığınız API anahtarı.

Kurallar:

- `=` işaretinin çevresine tırnak veya boşluk koymayın.
- Değerlerin başında/sonunda boşluk bırakmayın.
- `.env` Git tarafından yok sayılır; **asla commit edilmemeli, paylaşılmamalı veya
  ekran görüntüsüne alınmamalıdır.**

**Son kontrol listesi:**

- [ ] ElevenLabs hesabında kota/faturalandırma durumu uygun.
- [ ] Google Cloud projesinde faturalandırma (billing) bağlı.
- [ ] **Cloud Translation API** etkin.
- [ ] Google API anahtarı **Cloud Translation API** ile kısıtlı.
- [ ] `.env` içinde üç satırın tümü dolu ve köşeli parantez yok.

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
