document
  .getElementById("tweetForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const tweetText = document.getElementById("tweetInput").value;
    const response = await fetch("YOUR_API_ENDPOINT", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: tweetText }),
    });
    const data = await response.json();
    document.getElementById("result").innerText = data.result
      ? "Disaster-related"
      : "Not disaster-related";
  });
