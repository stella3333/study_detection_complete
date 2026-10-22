#include "WiFi.h"
#include "StreamIO.h"
#include "VideoStream.h"
#include "RTSP.h"
#include "NNFaceDetectionRecognition.h"
#include "VideoStreamOverlay.h"
#include "WiFiServer.h"
#include <WiFiUdp.h>
#include <NTPClient.h>

#define CHANNEL_VIDEO 0
#define CHANNEL_NN 3
#define NN_WIDTH 576
#define NN_HEIGHT 320
#define HTTP_PORT 80
#define FACE_SIGNAL_TIMEOUT_MS 2000UL
#define START_CONFIRMATION_MS 15000UL
#define ABSENCE_PAUSE_MS 30000UL
#define MIN_FACE_AREA_RATIO 0.01f
#define MIN_FACE_MARGIN_RATIO 0.03f

// Set these to the Wi-Fi network that the computer will also use.
char ssid[] = "猜猜我是誰";
char pass[] = "stella13333";

VideoSetting videoConfig(VIDEO_FHD, 30, VIDEO_H264, 0);
VideoSetting nnConfig(NN_WIDTH, NN_HEIGHT, 10, VIDEO_RGB, 0);
NNFaceDetectionRecognition faceRecognition;
RTSP rtsp;
StreamIO videoStreamer(1, 1);
StreamIO faceDetectionStreamer(1, 1);
WiFiServer dashboardServer(HTTP_PORT);
WiFiUDP ntpUdp;
NTPClient clockClient(ntpUdp, "pool.ntp.org", 8L * 60L * 60L);

volatile uint32_t lastFaceDetectedAtMs = 0;
uint32_t candidateSinceMs = 0;
uint32_t lastAccountedAtMs = 0;
uint32_t absenceSinceMs = 0;
uint32_t accumulatedStudyMs = 0;
uint32_t leaveCount = 0;
bool candidateActive = false;
bool studyStarted = false;
bool studyPaused = false;
bool leaveCountedForCurrentAbsence = false;
bool clockSynchronized = false;
const uint8_t MAX_LEAVE_HISTORY = 50;
struct LeaveRecord {
    uint32_t startedAtEpoch;
    uint32_t returnedAtEpoch;
    uint32_t startedAtMs;
    uint32_t durationSeconds;
    bool completed;
};
LeaveRecord leaveHistory[MAX_LEAVE_HISTORY] = {};
uint8_t leaveHistoryCount = 0;
int8_t activeLeaveIndex = -1;
uint32_t todayAwaySeconds = 0;
uint32_t todayDateKey = 0;
enum PomodoroPhase {
    POMODORO_FOCUS,
    POMODORO_BREAK
};
PomodoroPhase pomodoroPhase = POMODORO_FOCUS;
uint32_t focusDurationMs = 5UL * 60UL * 1000UL;
uint32_t breakDurationMs = 1UL * 60UL * 1000UL;
uint32_t pomodoroPhaseElapsedMs = 0;
uint32_t pomodoroPhaseDurationMs = focusDurationMs;
uint32_t pomodoroPhaseStartedAtMs = 0;
uint32_t pomodoroTransitionId = 0;

void faceResultCallback(std::vector<FaceRecognitionResult> results);
void updateStudyState(uint32_t now);
void handleDashboardClient(WiFiClient& client);
void sendHttpResponse(WiFiClient& client, int statusCode, const char* statusText, const char* contentType, const String& body);
const char* studyStateName(uint32_t now, bool facePresent, uint32_t& startRemainingMs);
bool parseQueryParameter(const String& requestLine, const char* name, uint32_t& value);
void accountFocusedTime(uint32_t elapsedMs, uint32_t now);
uint32_t dateKeyFromEpoch(uint32_t epoch);
uint32_t epochAtMillis(uint32_t timestampMs, uint32_t now);
void startLeaveRecord(uint32_t lastFaceAtMs, uint32_t now);
void finishLeaveRecord(uint32_t now);
void updateTodayDate();
String formatLocalTimestamp(uint32_t epoch);
uint32_t elapsedSince(uint32_t now, uint32_t timestamp);

void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println("BOOT,FOCUS_FACE_DETECTION_DEMO");

    int status = WL_IDLE_STATUS;
    while (status != WL_CONNECTED) {
        Serial.println("WIFI_CONNECTING");
        status = WiFi.begin(ssid, pass);
        delay(2000);
    }

    clockClient.begin();
    clockSynchronized = clockClient.forceUpdate();
    if (clockSynchronized) {
        updateTodayDate();
        Serial.println("CLOCK_SYNCED,TIMEZONE=UTC+8");
    } else {
        Serial.println("WARNING,CLOCK_SYNC_FAILED");
    }

    IPAddress ip = WiFi.localIP();
    Serial.print("WIFI_CONNECTED,IP=");
    Serial.println(ip);

    dashboardServer.begin();

    videoConfig.setBitrate(2 * 1024 * 1024);
    Camera.configVideoChannel(CHANNEL_VIDEO, videoConfig);
    Camera.configVideoChannel(CHANNEL_NN, nnConfig);
    Camera.videoInit();

    rtsp.configVideo(videoConfig);
    rtsp.begin();

    faceRecognition.configVideo(nnConfig);
    faceRecognition.modelSelect(FACE_RECOGNITION, NA_MODEL, DEFAULT_SCRFD, DEFAULT_MOBILEFACENET);
    faceRecognition.begin();
    faceRecognition.setResultCallback(faceResultCallback);

    videoStreamer.registerInput(Camera.getStream(CHANNEL_VIDEO));
    videoStreamer.registerOutput(rtsp);
    if (videoStreamer.begin() != 0) {
        Serial.println("ERROR,VIDEO_STREAM_START_FAILED");
    }

    faceDetectionStreamer.registerInput(Camera.getStream(CHANNEL_NN));
    faceDetectionStreamer.setStackSize();
    faceDetectionStreamer.setTaskPriority();
    faceDetectionStreamer.registerOutput(faceRecognition);
    if (faceDetectionStreamer.begin() != 0) {
        Serial.println("ERROR,FACE_DETECTION_STREAM_START_FAILED");
    }

    OSD.configVideo(CHANNEL_VIDEO, videoConfig);
    OSD.begin();

    Serial.print("RTSP_URL=rtsp://");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(rtsp.getPort());
    Serial.print("BOARD_API_URL=http://");
    Serial.print(ip);
    Serial.println("/");
    Serial.println("READY,OPEN_DASHBOARD_HTML_ON_COMPUTER");

    Camera.channelBegin(CHANNEL_VIDEO);
    Camera.channelBegin(CHANNEL_NN);
}

void loop()
{
    const uint32_t now = millis();
    static uint32_t lastNtpAttemptMs = now;
    if (now - lastNtpAttemptMs >= 60000UL) {
        lastNtpAttemptMs = now;
        if (clockClient.forceUpdate()) {
            clockSynchronized = true;
            updateTodayDate();
        } else {
            Serial.println("WARNING,CLOCK_SYNC_FAILED");
        }
    }
    updateStudyState(now);

    WiFiClient client = dashboardServer.available();
    if (client) {
        handleDashboardClient(client);
    }

    delay(10);
}

void faceResultCallback(std::vector<FaceRecognitionResult> results)
{
    const int faceCount = static_cast<int>(results.size());
    static int previousFaceCount = -1;
    static uint32_t lastReportMillis = 0;
    const uint32_t now = millis();

    int validFaceCount = 0;
    OSD.createBitmap(CHANNEL_VIDEO);
    const int imageWidth = videoConfig.width();
    const int imageHeight = videoConfig.height();

    for (int i = 0; i < faceCount; i++) {
        FaceRecognitionResult face = results[i];
        const int xMin = static_cast<int>(face.xMin() * imageWidth);
        const int xMax = static_cast<int>(face.xMax() * imageWidth);
        const int yMin = static_cast<int>(face.yMin() * imageHeight);
        const int yMax = static_cast<int>(face.yMax() * imageHeight);
        const int boxWidth = xMax - xMin;
        const int boxHeight = yMax - yMin;
        const float boxArea = (float)(boxWidth * boxHeight) / (float)(imageWidth * imageHeight);
        const bool isValidBox = boxWidth > 0 && boxHeight > 0 && boxArea >= MIN_FACE_AREA_RATIO;
        if (!isValidBox) {
            continue;
        }

        const uint32_t color = String(face.name()) == "unknown" ? OSD_COLOR_RED : OSD_COLOR_GREEN;
        OSD.drawRect(CHANNEL_VIDEO, xMin, yMin, xMax, yMax, 3, color);
        char label[40];
        snprintf(label, sizeof(label), "Face:%s", face.name());
        OSD.drawText(CHANNEL_VIDEO, xMin, yMin - OSD.getTextHeight(CHANNEL_VIDEO), label, color);
        validFaceCount++;
    }

    if (validFaceCount != previousFaceCount || now - lastReportMillis >= 1000) {
        if (validFaceCount > 0) {
            lastFaceDetectedAtMs = now;
            Serial.print("STATE=FACE_PRESENT,COUNT=");
            Serial.print(validFaceCount);
            Serial.print(",NAME=");
            Serial.println(results[0].name());
        } else {
            Serial.println("STATE=NO_FACE,COUNT=0");
        }
        previousFaceCount = validFaceCount;
        lastReportMillis = now;
    }

    // The SDK does not expose a clear() method on VideoStreamOverlay.
    // Recreating the bitmap at the start of each callback already resets the overlay,
    // so no valid face means we simply render nothing and update the channel.
    OSD.update(CHANNEL_VIDEO);
}

void updateStudyState(uint32_t now)
{
    uint32_t lastFaceAt = lastFaceDetectedAtMs;
    bool facePresent = lastFaceAt != 0 &&
        elapsedSince(now, lastFaceAt) <= FACE_SIGNAL_TIMEOUT_MS;

    if (!studyStarted) {
        if (facePresent) {
            if (!candidateActive) {
                candidateActive = true;
                candidateSinceMs = now;
            }
            if (now - candidateSinceMs >= START_CONFIRMATION_MS) {
                studyStarted = true;
                studyPaused = false;
                absenceSinceMs = 0;
                accumulatedStudyMs = 0;
                lastAccountedAtMs = now;
                pomodoroPhase = POMODORO_FOCUS;
                pomodoroPhaseElapsedMs = 0;
                pomodoroPhaseDurationMs = focusDurationMs;
                pomodoroPhaseStartedAtMs = now;
                pomodoroTransitionId++;
                Serial.println("TIMER=STARTED");
            }
        } else {
            candidateActive = false;
        }
        return;
    }

    if (pomodoroPhase == POMODORO_BREAK) {
        if (now - pomodoroPhaseStartedAtMs < pomodoroPhaseDurationMs) {
            lastAccountedAtMs = now;
            return;
        }

        pomodoroPhase = POMODORO_FOCUS;
        pomodoroPhaseElapsedMs = 0;
        pomodoroPhaseDurationMs = focusDurationMs;
        pomodoroPhaseStartedAtMs = now;
        lastAccountedAtMs = now;
        pomodoroTransitionId++;
        Serial.println("POMODORO=FOCUS_STARTED");
        lastFaceAt = lastFaceDetectedAtMs;
        facePresent = lastFaceAt != 0 &&
            elapsedSince(now, lastFaceAt) <= FACE_SIGNAL_TIMEOUT_MS;
        if (facePresent) {
            absenceSinceMs = 0;
        } else {
            lastFaceDetectedAtMs = 0;
            lastFaceAt = 0;
            absenceSinceMs = now;
        }
    }

    if (facePresent) {
        absenceSinceMs = 0;
        if (studyPaused) {
            finishLeaveRecord(now);
            studyPaused = false;
            leaveCountedForCurrentAbsence = false;
            lastAccountedAtMs = now;
            Serial.println("TIMER=RESUMED");
        } else {
            accountFocusedTime(now - lastAccountedAtMs, now);
            lastAccountedAtMs = now;
        }
        return;
    }

    const uint32_t absenceStartedAt = lastFaceAt != 0 ? lastFaceAt : absenceSinceMs;
    const uint32_t absenceAgeMs = elapsedSince(now, absenceStartedAt);
    const uint32_t pauseAtMs = absenceStartedAt + ABSENCE_PAUSE_MS;
    if (!studyPaused && absenceStartedAt != 0 && absenceAgeMs > ABSENCE_PAUSE_MS) {
        if (static_cast<int32_t>(pauseAtMs - lastAccountedAtMs) > 0) {
            accountFocusedTime(pauseAtMs - lastAccountedAtMs, pauseAtMs);
        }
        studyPaused = true;
        lastAccountedAtMs = now;
        if (!leaveCountedForCurrentAbsence) {
            leaveCount++;
            leaveCountedForCurrentAbsence = true;
            startLeaveRecord(absenceStartedAt, now);
        }
        Serial.print("TIMER=PAUSED,LEAVE_COUNT=");
        Serial.println(leaveCount);
    } else if (!studyPaused) {
        accountFocusedTime(now - lastAccountedAtMs, now);
        lastAccountedAtMs = now;
    }
}

void accountFocusedTime(uint32_t elapsedMs, uint32_t now)
{
    if (pomodoroPhase != POMODORO_FOCUS) {
        return;
    }

    accumulatedStudyMs += elapsedMs;
    pomodoroPhaseElapsedMs += elapsedMs;
    if (pomodoroPhaseElapsedMs >= pomodoroPhaseDurationMs) {
        pomodoroPhase = POMODORO_BREAK;
        pomodoroPhaseElapsedMs = 0;
        pomodoroPhaseDurationMs = breakDurationMs;
        pomodoroPhaseStartedAtMs = now;
        pomodoroTransitionId++;
        Serial.println("POMODORO=BREAK_STARTED");
    }
}

uint32_t dateKeyFromEpoch(uint32_t epoch)
{
    if (epoch == 0) {
        return 0;
    }

    time_t timestamp = static_cast<time_t>(epoch);
    struct tm* date = gmtime(&timestamp);
    if (date == NULL) {
        return 0;
    }
    return static_cast<uint32_t>((date->tm_year + 1900) * 10000 +
        (date->tm_mon + 1) * 100 + date->tm_mday);
}

uint32_t epochAtMillis(uint32_t timestampMs, uint32_t now)
{
    if (!clockSynchronized) {
        return 0;
    }
    return clockClient.getEpochTime() - elapsedSince(now, timestampMs) / 1000UL;
}

uint32_t elapsedSince(uint32_t now, uint32_t timestamp)
{
    const int32_t elapsed = static_cast<int32_t>(now - timestamp);
    return elapsed < 0 ? 0 : static_cast<uint32_t>(elapsed);
}

String formatLocalTimestamp(uint32_t epoch)
{
    if (epoch == 0) {
        return "";
    }

    time_t timestamp = static_cast<time_t>(epoch);
    struct tm* date = gmtime(&timestamp);
    if (date == NULL) {
        return "";
    }

    char formatted[24];
    strftime(formatted, sizeof(formatted), "%Y-%m-%d %H:%M:%S", date);
    return String(formatted);
}

void updateTodayDate()
{
    if (!clockSynchronized) {
        return;
    }

    const uint32_t currentDateKey = dateKeyFromEpoch(clockClient.getEpochTime());
    if (currentDateKey != 0 && currentDateKey != todayDateKey) {
        todayDateKey = currentDateKey;
        todayAwaySeconds = 0;
    }
}

void startLeaveRecord(uint32_t lastFaceAtMs, uint32_t now)
{
    if (leaveHistoryCount == MAX_LEAVE_HISTORY) {
        for (uint8_t i = 1; i < MAX_LEAVE_HISTORY; i++) {
            leaveHistory[i - 1] = leaveHistory[i];
        }
        leaveHistoryCount--;
        if (activeLeaveIndex >= 0) {
            activeLeaveIndex--;
        }
    }

    activeLeaveIndex = static_cast<int8_t>(leaveHistoryCount);
    LeaveRecord& record = leaveHistory[leaveHistoryCount++];
    record.startedAtEpoch = epochAtMillis(lastFaceAtMs, now);
    record.returnedAtEpoch = 0;
    record.startedAtMs = lastFaceAtMs;
    record.durationSeconds = elapsedSince(now, lastFaceAtMs) / 1000UL;
    record.completed = false;
}

void finishLeaveRecord(uint32_t now)
{
    if (activeLeaveIndex < 0 || activeLeaveIndex >= leaveHistoryCount) {
        return;
    }

    LeaveRecord& record = leaveHistory[activeLeaveIndex];
    record.completed = true;
    record.returnedAtEpoch = clockSynchronized ? clockClient.getEpochTime() : 0;
    if (record.startedAtEpoch == 0 && clockSynchronized) {
        record.startedAtEpoch = epochAtMillis(record.startedAtMs, now);
    }
    if (record.startedAtEpoch != 0 && record.returnedAtEpoch >= record.startedAtEpoch) {
        record.durationSeconds = record.returnedAtEpoch - record.startedAtEpoch;
        if (dateKeyFromEpoch(record.startedAtEpoch) == todayDateKey) {
            todayAwaySeconds += record.durationSeconds;
        }
    } else {
        record.durationSeconds = elapsedSince(now, record.startedAtMs) / 1000UL;
    }
    activeLeaveIndex = -1;
}

const char* studyStateName(uint32_t now, bool facePresent, uint32_t& startRemainingMs)
{
    startRemainingMs = 0;
    if (!studyStarted) {
        if (facePresent && candidateActive) {
            const uint32_t elapsed = now - candidateSinceMs;
            startRemainingMs = elapsed >= START_CONFIRMATION_MS ? 0 : START_CONFIRMATION_MS - elapsed;
            return "STARTING";
        }
        return "WAITING";
    }
    if (studyPaused) {
        return "PAUSED";
    }
    return facePresent ? "FOCUSING" : "AWAY_GRACE";
}

void handleDashboardClient(WiFiClient& client)
{
    client.setTimeout(250);
    String requestLine = client.readStringUntil('\n');
    requestLine.trim();

    while (client.connected()) {
        String headerLine = client.readStringUntil('\n');
        if (headerLine.length() <= 1) {
            break;
        }
    }

    if (requestLine.startsWith("OPTIONS ")) {
        sendHttpResponse(client, 204, "No Content", "text/plain; charset=utf-8", "");
    } else if (requestLine.startsWith("GET /api/settings?")) {
        uint32_t requestedFocusMinutes = 0;
        uint32_t requestedBreakMinutes = 0;
        const bool hasFocus = parseQueryParameter(requestLine, "focusMinutes", requestedFocusMinutes);
        const bool hasBreak = parseQueryParameter(requestLine, "breakMinutes", requestedBreakMinutes);
        if (!hasFocus || !hasBreak || requestedFocusMinutes < 1 || requestedFocusMinutes > 180 ||
            requestedBreakMinutes < 1 || requestedBreakMinutes > 60) {
            sendHttpResponse(client, 400, "Bad Request", "application/json; charset=utf-8",
                "{\"error\":\"focusMinutes must be 1-180 and breakMinutes must be 1-60\"}");
        } else {
            focusDurationMs = requestedFocusMinutes * 60UL * 1000UL;
            breakDurationMs = requestedBreakMinutes * 60UL * 1000UL;
            String body = "{\"focusMinutes\":";
            body += String(requestedFocusMinutes);
            body += ",\"breakMinutes\":";
            body += String(requestedBreakMinutes);
            body += "}";
            sendHttpResponse(client, 200, "OK", "application/json; charset=utf-8", body);
        }
    } else if (requestLine.startsWith("GET /api/history ")) {
        updateTodayDate();
        const uint32_t now = millis();
        const uint32_t currentEpoch = clockSynchronized ? clockClient.getEpochTime() : 0;
        uint32_t currentDayAwaySeconds = todayAwaySeconds;
        if (activeLeaveIndex >= 0 && activeLeaveIndex < leaveHistoryCount) {
            const LeaveRecord& activeRecord = leaveHistory[activeLeaveIndex];
            const uint32_t activeDurationSeconds = activeRecord.startedAtEpoch != 0 && currentEpoch >= activeRecord.startedAtEpoch
                ? currentEpoch - activeRecord.startedAtEpoch
                : elapsedSince(now, activeRecord.startedAtMs) / 1000UL;
            if (activeRecord.startedAtEpoch != 0 &&
                dateKeyFromEpoch(activeRecord.startedAtEpoch) == todayDateKey) {
                currentDayAwaySeconds += activeDurationSeconds;
            }
        }
        String body = "{\"timeSynced\":";
        body += clockSynchronized ? "true" : "false";
        body += ",\"date\":\"";
        body += formatLocalTimestamp(currentEpoch).substring(0, 10);
        body += "\",\"todayAwaySeconds\":";
        body += String(currentDayAwaySeconds);
        body += ",\"records\":[";
        for (uint8_t i = 0; i < leaveHistoryCount; i++) {
            if (i > 0) {
                body += ",";
            }
            LeaveRecord& record = leaveHistory[i];
            uint32_t durationSeconds = record.durationSeconds;
            if (!record.completed) {
                durationSeconds = record.startedAtEpoch != 0 && currentEpoch >= record.startedAtEpoch
                    ? currentEpoch - record.startedAtEpoch
                            : elapsedSince(now, record.startedAtMs) / 1000UL;
            }
            body += "{\"startedAt\":\"";
            body += formatLocalTimestamp(record.startedAtEpoch);
            body += "\",\"returnedAt\":\"";
            body += formatLocalTimestamp(record.returnedAtEpoch);
            body += "\",\"durationSeconds\":";
            body += String(durationSeconds);
            body += ",\"completed\":";
            body += record.completed ? "true" : "false";
            body += "}";
        }
        body += "]}";
        sendHttpResponse(client, 200, "OK", "application/json; charset=utf-8", body);
    } else if (requestLine.startsWith("GET /api/status ")) {
        const uint32_t now = millis();
        const uint32_t lastFaceAt = lastFaceDetectedAtMs;
        const bool facePresent = lastFaceAt != 0 &&
            elapsedSince(now, lastFaceAt) <= FACE_SIGNAL_TIMEOUT_MS;
        uint32_t startRemainingMs = 0;
        const char* state = studyStateName(now, facePresent, startRemainingMs);

        String body = "{\"state\":\"";
        body += state;
        body += "\",\"facePresent\":";
        body += facePresent ? "true" : "false";
        body += ",\"elapsedSeconds\":";
        body += String(accumulatedStudyMs / 1000UL);
        body += ",\"leaveCount\":";
        body += String(leaveCount);
        body += ",\"startRemainingSeconds\":";
        body += String((startRemainingMs + 999UL) / 1000UL);
        body += ",\"started\":";
        body += studyStarted ? "true" : "false";
        body += ",\"phase\":\"";
        body += pomodoroPhase == POMODORO_BREAK ? "BREAK" : "FOCUS";
        body += "\",\"phaseRemainingSeconds\":";
        const uint32_t phaseRemainingMs = pomodoroPhase == POMODORO_BREAK
            ? (now - pomodoroPhaseStartedAtMs >= pomodoroPhaseDurationMs ? 0 : pomodoroPhaseDurationMs - (now - pomodoroPhaseStartedAtMs))
            : (pomodoroPhaseElapsedMs >= pomodoroPhaseDurationMs ? 0 : pomodoroPhaseDurationMs - pomodoroPhaseElapsedMs);
        body += String((phaseRemainingMs + 999UL) / 1000UL);
        body += ",\"focusMinutes\":";
        body += String(focusDurationMs / 60000UL);
        body += ",\"breakMinutes\":";
        body += String(breakDurationMs / 60000UL);
        body += ",\"transitionId\":";
        body += String(pomodoroTransitionId);
        body += "}";
        sendHttpResponse(client, 200, "OK", "application/json; charset=utf-8", body);
    } else if (requestLine.startsWith("GET / ")) {
        sendHttpResponse(client, 200, "OK", "text/plain; charset=utf-8",
            "Open focus_detection_demo/dashboard.html on your computer and connect to this board's IP address.");
    } else {
        sendHttpResponse(client, 404, "Not Found", "text/plain; charset=utf-8", "Not found");
    }

    client.stop();
}

bool parseQueryParameter(const String& requestLine, const char* name, uint32_t& value)
{
    String key = String(name) + "=";
    const int keyIndex = requestLine.indexOf(key);
    if (keyIndex < 0) {
        return false;
    }

    const int valueStart = keyIndex + key.length();
    int valueEnd = valueStart;
    while (valueEnd < static_cast<int>(requestLine.length()) &&
           requestLine[valueEnd] >= '0' && requestLine[valueEnd] <= '9') {
        valueEnd++;
    }
    if (valueEnd == valueStart || (requestLine[valueEnd] != '&' && requestLine[valueEnd] != ' ')) {
        return false;
    }

    value = static_cast<uint32_t>(requestLine.substring(valueStart, valueEnd).toInt());
    return true;
}

void sendHttpResponse(WiFiClient& client, int statusCode, const char* statusText, const char* contentType, const String& body)
{
    client.print("HTTP/1.1 ");
    client.print(statusCode);
    client.print(" ");
    client.println(statusText);
    client.print("Content-Type: ");
    client.println(contentType);
    client.println("Access-Control-Allow-Origin: *");
    client.println("Access-Control-Allow-Methods: GET, OPTIONS");
    client.println("Access-Control-Allow-Headers: *");
    client.println("Access-Control-Allow-Private-Network: true");
    client.println("Cache-Control: no-store");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(body.length());
    client.println();
    client.print(body);
}
