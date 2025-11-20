document.getElementById("uploadBtn").addEventListener("click", async () => {

    const videoFile = document.getElementById("videoFile").files[0];
    const csvFile   = document.getElementById("csvFile").files[0];

    if (!videoFile || !csvFile) {
        alert("영상과 CSV 파일을 모두 선택하세요.");
        return;
    }

    const statusBox = document.getElementById("statusBox");
    statusBox.textContent = "파일 업로드 중...";

    let formData = new FormData();
    formData.append("video", videoFile);
    formData.append("telemetry", csvFile);

    // ---------- 1) 업로드 ----------
    const uploadResponse = await fetch("/api/upload", {
        method: "POST",
        body: formData
    });

    const uploadData = await uploadResponse.json();
    console.log(uploadData);

    if (!uploadData.success) {
        statusBox.textContent = "업로드 실패: " + uploadData.error;
        return;
    }

    const upload_id = uploadData.upload_id;

    statusBox.textContent = "분석 중... (20~40초 소요)";

    // ---------- 2) 분석 ----------
    const analyzeResponse = await fetch("/api/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ upload_id: upload_id })
    });

    const result = await analyzeResponse.json();
    console.log(result);

    if (!result.success) {
        statusBox.textContent = "분석 실패: " + result.error;
        return;
    }

    statusBox.textContent = "완료!";

    document.getElementById("resultBox").innerHTML = `
        <h2>분석 결과</h2>
        <p style="color:#ff5500;">오버레이 영상 생성됨</p>
        <a href="/outputs/${result.output_video}" style="color:#fff;">[다운로드]</a>
        
        <h3>LAP 분석</h3>
        <pre>${JSON.stringify(result.laps, null, 2)}</pre>

        <h3>성능 분석</h3>
        <pre>${JSON.stringify(result.performance, null, 2)}</pre>

        <h3>AI 피드백</h3>
        <pre>${JSON.stringify(result.feedback, null, 2)}</pre>
    `;
});
