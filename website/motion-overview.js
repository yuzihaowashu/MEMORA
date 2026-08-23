const scenes = [
  { duration: null, caption: "Experience unfolds in 10-second segments." },
  { duration: null, caption: "Layered observations enter typed memory." },
  { duration: null, caption: "Memory changes as entities change." },
  { duration: 3200, caption: "Repeated episodes become reusable knowledge." },
  { duration: 3600, caption: "A new goal retrieves what matters." },
  { duration: 6000, caption: "Memory grounds a participant-specific plan." },
  { duration: 17200, caption: "The plan crosses into physical execution." }
];

const pageParams = new URLSearchParams(window.location.search);
const captureMode = pageParams.get("capture") === "1";

if (pageParams.get("render") === "hd") {
  document.body.classList.add("render-hd");
} else if (pageParams.get("render") === "retina") {
  document.body.classList.add("render-retina");
}

const humanVideo = document.getElementById("human-video");
const robotVideo = document.getElementById("robot-video");
const caption = document.getElementById("scene-caption");
const progress = [...document.querySelectorAll(".scene-progress span")];
const segmentLabels = [...document.querySelectorAll(".segment-track span")];
const segmentTime = document.getElementById("segment-time");
const observationLine = document.querySelector("#observation-line strong");
const memoryStatus = document.getElementById("memory-status");

let sceneIndex = 0;
let sceneTimer;

function updateSegmentFromVideo() {
  const time = humanVideo.currentTime;
  const segment = time < 10 ? 0 : time < 20 ? 1 : 2;
  segmentLabels.forEach((label, index) => label.classList.toggle("active", index === segment));
  segmentTime.textContent = ["00:00–00:10", "00:10–00:20", "00:20–00:22"][segment];
  observationLine.textContent = [
    "Breakfast items appear in a shared workspace.",
    "The yellow bowl, tea packet, spoon, and bread change state.",
    "The assembled breakfast setup becomes persistent evidence."
  ][segment];

  if (sceneIndex === 0 && time >= 10) {
    showScene(1);
  } else if (sceneIndex === 1 && time >= 20) {
    showScene(2);
  } else if (sceneIndex === 2 && humanVideo.duration && time >= humanVideo.duration - 0.12) {
    showScene(3);
  }
}

humanVideo.addEventListener("timeupdate", updateSegmentFromVideo);
humanVideo.addEventListener("seeked", updateSegmentFromVideo);
humanVideo.addEventListener("ended", () => {
  updateSegmentFromVideo();
  if (sceneIndex === 2) showScene(3);
});

function advanceScene(index) {
  if (index === scenes.length - 1 && captureMode) {
    document.body.dataset.captureComplete = "1";
    robotVideo.pause();
    return;
  }
  showScene((index + 1) % scenes.length);
}

function showScene(index) {
  sceneIndex = index;
  document.body.dataset.scene = String(index);
  caption.textContent = scenes[index].caption;
  progress.forEach((item, itemIndex) => item.classList.toggle("active", itemIndex <= index));
  memoryStatus.textContent = index === 0 ? "Waiting for experience" : index === 1 ? "Encoding embodied experience" : index === 2 ? "Applying meaningful edits" : index === 3 ? "Consolidating across episodes" : "Available for retrieval";
  if (index === 0) {
    humanVideo.currentTime = 0;
    updateSegmentFromVideo();
    humanVideo.play().catch(() => {});
    robotVideo.pause();
    robotVideo.currentTime = 0;
  }
  if (index === 6) {
    humanVideo.pause();
    robotVideo.currentTime = 0;
    robotVideo.play().catch(() => {});
  }

  window.clearTimeout(sceneTimer);
  if (index === 2 && captureMode) {
    sceneTimer = window.setTimeout(() => showScene(3), 2300);
  } else if (scenes[index].duration !== null) {
    sceneTimer = window.setTimeout(() => advanceScene(index), scenes[index].duration);
  }
}

document.getElementById("replay-film").addEventListener("click", () => showScene(0));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(sceneTimer);
    humanVideo.pause();
    robotVideo.pause();
  } else {
    showScene(sceneIndex);
  }
});

if (captureMode) {
  window.startMotionCapture = () => showScene(0);
} else {
  showScene(0);
}
