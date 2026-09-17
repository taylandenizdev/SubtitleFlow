"use strict";

(function () {
  var state = {
    csrf: "",
    busy: false,
    pollTimer: null,
    openVideoId: "",
    hasSource: false,
    hasTranslation: false,
    activeKind: "source",
    loadSeq: 0,
  };

  var els = {
    form: document.getElementById("transcript-form"),
    url: document.getElementById("video-url"),
    urlError: document.getElementById("url-error"),
    allowPaid: document.getElementById("allow-paid"),
    burnVideo: document.getElementById("burn-video"),
    reservation: document.getElementById("reservation"),
    perCallReservation: document.getElementById("per-call-reservation"),
    sourceLanguage: document.getElementById("source-language"),
    submit: document.getElementById("submit"),
    status: document.getElementById("status"),
    result: document.getElementById("result"),
    resultTitle: document.getElementById("result-title"),
    resultFile: document.getElementById("result-file"),
    resultVideo: document.getElementById("result-video"),
    resultReveal: document.getElementById("result-reveal"),
    videoReveal: document.getElementById("video-reveal"),
    resultTranslate: document.getElementById("result-translate"),
    tabSource: document.getElementById("tab-source"),
    tabTranslation: document.getElementById("tab-translation"),
    viewSource: document.getElementById("view-source"),
    viewTranslation: document.getElementById("view-translation"),
    sourceDownload: document.getElementById("source-download"),
    sourceSave: document.getElementById("source-save"),
    sourceCopy: document.getElementById("source-copy"),
    sourcePreview: document.getElementById("source-preview"),
    translationDownload: document.getElementById("translation-download"),
    translationSave: document.getElementById("translation-save"),
    translationCopy: document.getElementById("translation-copy"),
    translationPreview: document.getElementById("translation-preview"),
    library: document.getElementById("library"),
    libraryReveal: document.getElementById("library-reveal"),
    libraryList: document.getElementById("library-list"),
    runtimeNote: document.getElementById("runtime-note"),
  };

  // The narrow Python bridge is only present inside the desktop window. The
  // browser route keeps its plain download links and stays fully usable.
  function desktopApi() {
    var bridge = window.pywebview;
    if (
      bridge &&
      bridge.api &&
      typeof bridge.api.save_transcript === "function" &&
      typeof bridge.api.open_transcripts_folder === "function"
    ) {
      return bridge.api;
    }
    return null;
  }

  // Folder/save actions only exist behind the native bridge. In an ordinary
  // browser the buttons stay hidden so no dead control is shown.
  function applyRuntimeSurface() {
    var isDesktop = desktopApi() !== null;
    if (els.runtimeNote) {
      els.runtimeNote.textContent = isDesktop
        ? "Masaüstü penceresinde çalışır."
        : "Tarayıcıda çalışır.";
    }
    if (els.libraryReveal) {
      els.libraryReveal.hidden = !isDesktop;
    }
    if (els.resultReveal) {
      els.resultReveal.hidden = !isDesktop;
    }
    if (els.sourceSave) {
      els.sourceSave.hidden = !isDesktop;
    }
    if (els.translationSave) {
      els.translationSave.hidden = !isDesktop;
    }
    if (els.videoReveal && els.resultVideo) {
      var hasVideo = els.resultVideo.hidden === false;
      els.videoReveal.hidden = !(hasVideo && isDesktop);
    }
  }

  function setStatus(text, kind) {
    els.status.textContent = text;
    if (kind) {
      els.status.setAttribute("data-state", kind);
    } else {
      els.status.removeAttribute("data-state");
    }
  }

  function showUrlError(message) {
    els.urlError.textContent = message;
    els.urlError.hidden = false;
    els.url.setAttribute("aria-invalid", "true");
  }

  function clearUrlError() {
    els.urlError.textContent = "";
    els.urlError.hidden = true;
    els.url.removeAttribute("aria-invalid");
  }

  function validateUrl(value) {
    var text = (value || "").trim();
    if (text === "") {
      return { error: "Bir video bağlantısı girin." };
    }
    var parsed;
    try {
      parsed = new URL(text);
    } catch (err) {
      return { error: "Bağlantı geçerli bir adres değil." };
    }
    if (parsed.protocol !== "https:") {
      return { error: "Yalnız https bağlantıları kabul edilir." };
    }
    var host = parsed.hostname.toLowerCase();
    var allowed = [
      "youtube.com",
      "www.youtube.com",
      "m.youtube.com",
      "youtu.be",
    ];
    if (allowed.indexOf(host) === -1) {
      return { error: "Yalnız YouTube video bağlantıları kabul edilir." };
    }
    return { value: text };
  }

  function toggleCloudOptions() {
    // Every dispatch is paid (Scribe STT and Google Translation LLM), so the
    // shared paid-consent/reservation controls are always available.
    var nodes = document.querySelectorAll(".cloud-option");
    for (var i = 0; i < nodes.length; i += 1) {
      nodes[i].hidden = false;
    }
  }

  function guideToPaidConsent(message) {
    if (
      typeof message === "string" &&
      (message.indexOf("kullanımına izin ver") !== -1 ||
        message.indexOf("Google Translation LLM") !== -1)
    ) {
      if (els.allowPaid && !els.allowPaid.hidden) {
        els.allowPaid.focus();
      }
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    els.submit.disabled = busy;
  }

  function applyBootstrap(data) {
    state.csrf = data.csrf_token || "";
    if (els.reservation && !els.reservation.value) {
      els.reservation.value =
        data.default_total_reservation || data.default_reservation || "1.00";
    }
    if (els.perCallReservation && !els.perCallReservation.value) {
      els.perCallReservation.value =
        data.default_per_call_reservation || "0.05";
    }
    applyGoogleMtSupport(data.google_translation_configured === true);
    renderLibrary(data.library || data.transcripts || []);
  }

  // The fixed MT route needs a configured project/API key. Only a configured
  // boolean arrives from the server; the key itself never does. When it is not
  // configured, the operator sees an actionable hint before any dispatch.
  function applyGoogleMtSupport(configured) {
    var hint = document.getElementById("mt-google-hint");
    if (!hint) {
      return;
    }
    if (configured) {
      hint.hidden = true;
      hint.textContent = "";
      return;
    }
    hint.hidden = false;
    hint.textContent =
      "Google Translation LLM yapılandırılmamış: .env içinde " +
      "GOOGLE_TRANSLATION_PROJECT ve GOOGLE_TRANSLATION_API_KEY " +
      "tanımlanmalıdır. Anahtar arayüze girilmez.";
  }

  function formatSize(bytes) {
    if (bytes < 1024) {
      return bytes + " B";
    }
    return (bytes / 1024).toFixed(1) + " KB";
  }

  function renderLibrary(items) {
    els.libraryList.textContent = "";
    if (!items.length) {
      els.library.hidden = true;
      return;
    }
    els.library.hidden = false;
    items.forEach(function (item) {
      var li = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.className = "library-item";

      var text = document.createElement("span");
      text.className = "library-text";
      var title = document.createElement("span");
      title.className = "library-title";
      title.textContent = item.title || item.video_id;
      var meta = document.createElement("span");
      meta.className = "library-meta";
      var badges = [];
      if (item.source) {
        badges.push("Kaynak");
      }
      if (item.translation) {
        badges.push("Türkçe");
      }
      meta.textContent = badges.join(" · ") + " · " + formatSize(item.size_bytes);
      text.appendChild(title);
      text.appendChild(meta);
      button.appendChild(text);
      button.addEventListener("click", function () {
        showStored(item.video_id, item);
      });

      var translate = document.createElement("button");
      translate.type = "button";
      translate.className = "library-translate";
      if (item.translation) {
        // View the existing translation; never imply a silent regeneration.
        translate.textContent = "Türkçe çeviriyi aç";
        translate.addEventListener("click", function (event) {
          event.stopPropagation();
          showStored(item.video_id, item);
        });
      } else if (item.translatable === false) {
        translate.textContent = "Arşiv kaynağı yok";
        translate.disabled = true;
        translate.title =
          "Bu belge için arşivlenmiş iş kanıtı yok; Türkçe çeviri yalnız " +
          "doğrulanmış arşiv kaynağıyla çalışır. Belgeyi görüntüleyebilirsiniz.";
      } else {
        translate.textContent = "Türkçeye çevir";
        translate.addEventListener("click", function (event) {
          event.stopPropagation();
          startTranslation(item.video_id);
        });
      }

      li.appendChild(button);
      li.appendChild(translate);
      els.libraryList.appendChild(li);
    });
  }

  function safeVideoId(value) {
    return typeof value === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(value);
  }

  function fetchDocument(kind, videoId) {
    var prefix = kind === "translation" ? "/api/translation/" : "/api/source/";
    return fetch(prefix + encodeURIComponent(videoId), {
      headers: { Accept: "application/json" },
    }).then(function (response) {
      if (!response.ok) {
        return null;
      }
      return response.json();
    });
  }

  function setActiveTab(kind) {
    var source = kind !== "translation";
    state.activeKind = source ? "source" : "translation";
    els.tabSource.setAttribute("aria-selected", source ? "true" : "false");
    els.tabTranslation.setAttribute("aria-selected", source ? "false" : "true");
    els.tabSource.classList.toggle("tab-active", source);
    els.tabTranslation.classList.toggle("tab-active", !source);
    els.viewSource.hidden = !source;
    els.viewTranslation.hidden = source;
    if (els.resultReveal) {
      els.resultReveal.textContent = source
        ? "Kaynak klasörünü aç"
        : "Türkçe klasörünü aç";
    }
  }

  function renderDocuments(hasSource, hasTranslation) {
    state.hasSource = hasSource;
    state.hasTranslation = hasTranslation;
    els.tabSource.hidden = !hasSource;
    els.tabTranslation.hidden = !hasTranslation;
    els.tabTranslation.classList.toggle("needs-help", hasSource && !hasTranslation);
    if (hasTranslation) {
      setActiveTab("translation");
    } else {
      setActiveTab("source");
    }
    if (els.resultTranslate) {
      els.resultTranslate.hidden = !(hasSource && !hasTranslation);
    }
    if (!hasSource && !hasTranslation) {
      els.result.hidden = true;
    }
  }

  // The burned-in video line is only shown when the finished job really bound a
  // published MP4; its folder action stays behind the native desktop bridge so
  // no dead control appears in an ordinary browser.
  function renderVideo(job) {
    var name =
      job && typeof job.video_file_name === "string" ? job.video_file_name : "";
    var hasVideo = name !== "";
    if (els.resultVideo) {
      els.resultVideo.hidden = !hasVideo;
      els.resultVideo.textContent = hasVideo ? "Altyazılı video: " + name : "";
    }
    if (els.videoReveal) {
      els.videoReveal.hidden = !(hasVideo && desktopApi() !== null);
    }
    if (hasVideo) {
      els.result.hidden = false;
    }
  }

  function showResultFallback() {
    if (state.hasTranslation) {
      setActiveTab("translation");
    } else if (state.hasSource) {
      setActiveTab("source");
    }
    els.result.hidden = false;
    els.result.scrollIntoView({ block: "nearest" });
  }

  function loadDocuments(videoId) {
    state.openVideoId = videoId;
    var seq = ++state.loadSeq;
    Promise.all([
      fetchDocument("source", videoId),
      fetchDocument("translation", videoId),
    ]).then(function (results) {
      // A slower earlier request must never repaint a newer selection or leave
      // a stale download target behind.
      if (seq !== state.loadSeq) {
        return;
      }
      var source = results[0];
      var translation = results[1];
      var hasSource = source !== null;
      var hasTranslation = translation !== null;

      els.result.hidden = false;
      els.resultTitle.textContent = hasTranslation
        ? "Kaynak ve Türkçe çeviri"
        : hasSource
        ? "Kaynak transkript hazır"
        : "Belge bulunamadı";
      els.resultFile.textContent = videoId + ".md";

      if (source) {
        els.sourcePreview.textContent = source.content || "";
        els.sourceDownload.hidden = false;
        els.sourceDownload.href =
          "/api/source/" + encodeURIComponent(videoId) + "/download";
        els.sourceDownload.setAttribute("download", videoId + ".md");
      } else {
        els.sourcePreview.textContent = "";
        els.sourceDownload.hidden = true;
        els.sourceDownload.removeAttribute("href");
      }
      if (translation) {
        els.translationPreview.textContent = translation.content || "";
        els.translationDownload.hidden = false;
        els.translationDownload.href =
          "/api/translation/" + encodeURIComponent(videoId) + "/download";
        els.translationDownload.setAttribute("download", videoId + ".md");
      } else {
        els.translationPreview.textContent =
          "Bu video için henüz Türkçe çeviri yok.";
        els.translationDownload.hidden = true;
        els.translationDownload.removeAttribute("href");
      }
      renderDocuments(hasSource, hasTranslation);
      els.result.scrollIntoView({ block: "nearest" });
    });
  }

  function showStored(videoId, item) {
    if (!safeVideoId(videoId)) {
      return;
    }
    loadDocuments(videoId);
  }

  function refreshLibrary() {
    fetch("/api/library", { headers: { Accept: "application/json" } })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        renderLibrary(data.library || data.transcripts || []);
      })
      .catch(function () {
        /* keep the current list on a transient failure */
      });
  }

  function finishJob(videoId) {
    setBusy(false);
    refreshLibrary();
    loadDocuments(videoId);
  }

  function pollJob(jobId, videoId, attempts) {
    fetch("/api/jobs/" + encodeURIComponent(jobId), {
      headers: { Accept: "application/json" },
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("job");
        }
        return response.json();
      })
      .then(function (job) {
        if (job.state === "done") {
          setStatus(
            job.message || "Hazır.",
            job.needs_review ? "review" : "done"
          );
          renderVideo(job);
          finishJob(job.video_id || videoId);
          return;
        }
        if (job.state === "partial") {
          setStatus(
            job.message || "Kaynak hazır; çeviri tamamlanamadı.",
            "partial"
          );
          renderVideo(job);
          finishJob(job.video_id || videoId);
          return;
        }
        if (job.state === "error") {
          setBusy(false);
          setStatus(job.message || "İş başarısız.", "error");
          renderVideo(job);
          guideToPaidConsent(job.message);
          // Keep any previously saved document listed for viewing.
          refreshLibrary();
          return;
        }
        if (job.stage) {
          setStatus(job.stage, "running");
        }
        state.pollTimer = window.setTimeout(function () {
          pollJob(jobId, videoId, attempts + 1);
        }, 1000);
      })
      .catch(function () {
        if (attempts < 10) {
          state.pollTimer = window.setTimeout(function () {
            pollJob(jobId, videoId, attempts + 1);
          }, 1500);
        } else {
          setBusy(false);
          setStatus("İş durumu alınamadı; sayfayı yenileyin.", "error");
        }
      });
  }

  function startJob(path, payload, videoId, runningText) {
    setBusy(true);
    setStatus(runningText || "İşleniyor…", "running");
    return fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": state.csrf,
      },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, status: response.status, data: data };
        });
      })
      .then(function (result) {
        if (!result.ok) {
          setBusy(false);
          var message =
            result.data && result.data.error
              ? result.data.error.message
              : "İş başlatılamadı.";
          if (result.status === 409) {
            setStatus(message, "error");
          } else {
            showUrlError(message);
            setStatus("Hazır");
          }
          return;
        }
        state.pollTimer = window.setTimeout(function () {
          pollJob(result.data.job_id, videoId, 0);
        }, 600);
      })
      .catch(function () {
        setBusy(false);
        setStatus("Sunucuya ulaşılamadı; yeniden deneyin.", "error");
      });
  }

  function submit(event) {
    event.preventDefault();
    if (state.busy) {
      return;
    }
    clearUrlError();
    var check = validateUrl(els.url.value);
    if (check.error) {
      showUrlError(check.error);
      els.url.focus();
      return;
    }

    var burnVideo = els.burnVideo ? els.burnVideo.checked === true : false;
    var payload = {
      url: check.value,
      allow_paid: els.allowPaid.checked,
      reservation: els.reservation.value,
      per_call_reservation: els.perCallReservation
        ? els.perCallReservation.value
        : null,
      source_language: els.sourceLanguage.value || null,
      burn_video: burnVideo,
    };
    renderVideo(null);
    var runningText = burnVideo
      ? "Kaynak alınıyor, Türkçeye çevriliyor ve altyazılı video hazırlanıyor…"
      : "Kaynak alınıyor ve Türkçeye çevriliyor…";
    startJob("/api/transcripts", payload, "", runningText);
  }

  function startTranslation(videoId) {
    if (state.busy || !safeVideoId(videoId)) {
      return;
    }
    startJob(
      "/api/translations",
      {
        video_id: videoId,
        source_language: els.sourceLanguage.value || null,
        allow_paid: els.allowPaid.checked,
        reservation: els.reservation.value,
        per_call_reservation: els.perCallReservation
          ? els.perCallReservation.value
          : null,
      },
      videoId,
      "Türkçe çeviri çalışıyor…"
    );
  }

  function saveDocument(videoId, kind) {
    var api = desktopApi();
    if (!api) {
      return false;
    }
    setStatus("Kaydediliyor…", "running");
    api.save_transcript(videoId, kind).then(function (result) {
      if (result && result.ok) {
        setStatus("Kaydedildi: " + (result.path || ""), "done");
      } else if (result && result.cancelled) {
        setStatus("Kaydetme iptal edildi.");
      } else {
        setStatus(
          (result && result.message) || "Dosya kaydedilemedi.",
          "error"
        );
      }
    });
    return true;
  }

  function copyDocument(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text || "").then(
        function () {
          setStatus("Metin kopyalandı.", "done");
        },
        function () {
          setStatus("Kopyalama başarısız.", "error");
        }
      );
    } else {
      setStatus("Kopyalama bu ortamda desteklenmiyor.", "error");
    }
  }

  function revealFolder(kind) {
    var api = desktopApi();
    if (!api) {
      setStatus("Klasör yalnız masaüstü penceresinde açılabilir.");
      return;
    }
    var resolved = "source";
    if (kind === "translation") {
      resolved = "translation";
    } else if (kind === "video") {
      resolved = "video";
    }
    api.open_transcripts_folder(resolved).then(function (result) {
      if (result && result.ok) {
        var label =
          resolved === "translation"
            ? "Türkçe klasörü açıldı."
            : resolved === "video"
            ? "Videolar klasörü açıldı."
            : "Kaynak klasörü açıldı.";
        setStatus(label, "done");
      } else {
        setStatus((result && result.message) || "Klasör açılamadı.", "error");
      }
    });
  }

  // Hook used by the native window to explain a vetoed close while a job runs.
  window.transkriptDesktop = {
    busyClose: function () {
      setStatus(
        "Bir iş sürüyor; pencereyi kapatmak için işin bitmesini bekleyin.",
        "running"
      );
      if (els.status && els.status.scrollIntoView) {
        els.status.scrollIntoView({ block: "nearest" });
      }
    },
  };

  els.form.addEventListener("submit", submit);
  els.url.addEventListener("input", clearUrlError);

  els.tabSource.addEventListener("click", function () {
    setActiveTab("source");
  });
  els.tabTranslation.addEventListener("click", function () {
    setActiveTab("translation");
  });

  els.sourceDownload.addEventListener("click", function (event) {
    if (state.openVideoId && desktopApi()) {
      event.preventDefault();
      saveDocument(state.openVideoId, "source");
    }
  });
  els.translationDownload.addEventListener("click", function (event) {
    if (state.openVideoId && desktopApi()) {
      event.preventDefault();
      saveDocument(state.openVideoId, "translation");
    }
  });
  els.sourceSave.addEventListener("click", function () {
    if (state.openVideoId) {
      saveDocument(state.openVideoId, "source");
    }
  });
  els.translationSave.addEventListener("click", function () {
    if (state.openVideoId) {
      saveDocument(state.openVideoId, "translation");
    }
  });
  els.sourceCopy.addEventListener("click", function () {
    copyDocument(els.sourcePreview.textContent);
  });
  els.translationCopy.addEventListener("click", function () {
    copyDocument(els.translationPreview.textContent);
  });
  els.resultTranslate.addEventListener("click", function () {
    if (state.openVideoId) {
      startTranslation(state.openVideoId);
    }
  });

  if (els.resultReveal) {
    els.resultReveal.addEventListener("click", function () {
      revealFolder(state.activeKind);
    });
  }
  if (els.videoReveal) {
    els.videoReveal.addEventListener("click", function () {
      revealFolder("video");
    });
  }
  if (els.libraryReveal) {
    els.libraryReveal.addEventListener("click", function () {
      revealFolder("source");
    });
  }

  toggleCloudOptions();
  setActiveTab("source");

  // pywebview injects its bridge after the page loads; re-apply when it arrives.
  applyRuntimeSurface();
  window.addEventListener("pywebviewready", applyRuntimeSurface);

  fetch("/api/bootstrap", { headers: { Accept: "application/json" } })
    .then(function (response) {
      if (!response.ok) {
        throw new Error("bootstrap");
      }
      return response.json();
    })
    .then(applyBootstrap)
    .catch(function () {
      setStatus("Arayüz yüklenemedi; sayfayı yenileyin.", "error");
    });
})();
