package com.jjassistant;

import android.app.Service;
import android.app.Notification;
import android.media.AudioManager;
import android.media.AudioDeviceInfo;
import android.content.Context;
import android.content.Intent;
import android.content.ComponentName;
import android.content.res.AssetManager;
import android.content.pm.ServiceInfo;
import android.os.Bundle;
import android.os.IBinder;
import android.os.Build;
import android.util.Log;

import androidx.core.app.NotificationCompat;
import androidx.annotation.Nullable;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;

import java.util.Date;
import java.util.concurrent.TimeUnit;
import java.util.HashMap;
import java.util.Map;
import java.lang.Thread;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.FileReader;
import java.io.FileInputStream;
import java.io.OutputStreamWriter;

class OpenWakeWordOptionsExample {
    public String model = null;
    public String threshold = null;
    public String trigger_level = null;
    public String refractory = null;
    public String step_frames = null;
    public String melspectrogram_model = null;
    public String embedding_model = null;
    public String debug = null;
    public boolean end_after_activation = false;
}

// backgound service example: https://gist.github.com/varunon9/f2beec0a743c96708eb0ef971a9ff9cd?permalink_comment_id=3831303
// learning wake word: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing

public class OpenWakeWordServiceExample extends Service {
    static { System.loadLibrary("openWakeWord"); }

    private static AssetManager mgr;

    //   0 - app is starting ...
    // * 1 - app is runned                      // _STARTED
    //   2 - app is stopping ..
    //   21 - waking or  cpp end ...
    // * 22 - waking and cpp end                // _STOPPED
    //   23 - cpp starting ..
    //   3 - closing strems (after cpp end) ...
    // * 4 - app is dead                        // _ENDED
    private static Number lifeCycle = 4;
    private static Boolean endApp = false;
    private static Boolean stopApp = false;

    private static boolean closeServiceAfterWakeWordActivation = false;
    private static int deviceId = 0;
    private static OpenWakeWordOptionsExample opts = new OpenWakeWordOptionsExample();
    private static String fifoOutFileName;
    private static String fifoInFileName;
    private static WorkManager worker;
    private static String old_requestID = "";

    public static String intentFilterBroadcastString;
    public static String workerName = "JJPluginWakeWordServiceRestertWorker";

    public native void openWakeWord(AssetManager mgr, OpenWakeWordOptionsExample opts, int deviceId, String fifoInFileName, String fifoOutFileName);
    public static native void endOpenWakeWord();

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d("~= OpenWakeWordService", "onStartCommand() lifeCycle: " + lifeCycle);

        Bundle extras = intent.getExtras();
        if (extras == null) return Service.START_REDELIVER_INTENT;

        String requestID = extras.getString("requestID");
        intentFilterBroadcastString = extras.getString("intentFilterBroadcastString", intentFilterBroadcastString);

        if (requestID.equals(old_requestID)) {
            return Service.START_REDELIVER_INTENT;
        } else old_requestID = requestID;

        if (extras.getString("end") != null && lifeCycle.equals(1)) {
            lifeCycle = 2;
            endApp = true;
            endOpenWakeWord();
            return Service.START_REDELIVER_INTENT;
        }
        else if (extras.getString("end") != null && lifeCycle.equals(22)) {
            lifeCycle = 3;
            return Service.START_REDELIVER_INTENT;
        }
        else if (extras.getString("end") != null && lifeCycle.equals(4)) {
            callback("_ENDED");
            return Service.START_REDELIVER_INTENT;
        }
        else if (extras.getString("stop") != null && lifeCycle.equals(22)) {
            callback("_STOPPED");
            return Service.START_REDELIVER_INTENT;
        }
        else if (extras.getString("stop") != null && lifeCycle.equals(1)) {
            lifeCycle = 2;
            stopApp = true;
            endOpenWakeWord();
            return Service.START_REDELIVER_INTENT;
        }
        else if (lifeCycle.equals(1)) {
            callback("_STARTED");
            return Service.START_REDELIVER_INTENT;
        }
        else if (lifeCycle.equals(22)) {
            lifeCycle = 23;
            stopApp = false;
            cppStart(extras, Integer.valueOf(extras.getString("delayMS", "0")));
            return Service.START_REDELIVER_INTENT;
        }
        else if (lifeCycle.equals(4) && extras.getString("keyword") != null) {
            Log.d("~= OpenWakeWordService", "STARTING");

            lifeCycle = 0;
            endApp = false;
            stopApp = false;

            closeServiceAfterWakeWordActivation = Boolean.parseBoolean(extras.getString("closeServiceAfterWakeWordActivation", "false"));

            File dir = getFilesDir();
            if(!dir.exists()) dir.mkdir();
            fifoOutFileName = getFilesDir() + "/fifoOut";
            fifoInFileName = getFilesDir() + "/fifoIn";

            NotificationCompat.Builder notification = new NotificationCompat.Builder(this, intentFilterBroadcastString)
                .setAutoCancel(false)
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle("JJAssistant")
                .setContentText("JJAssistant Vás počúva na pozadí")
                .setPriority(NotificationCompat.PRIORITY_DEFAULT);
            int type = 0;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                type = ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE;
            }
            startForeground(99, notification.build(), type);

            AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            AudioDeviceInfo[] devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS);
            for (AudioDeviceInfo device : devices) {
                if (AudioDeviceInfo.TYPE_BUILTIN_MIC == device.getType()) {
                    deviceId = device.getId();
                    break;
                }
            }

            new File(fifoOutFileName).delete();

            // stdout reader and callbeck
            new Thread(new Runnable() {
                @Override
                public void run() {
                    BufferedReader buffer = null;
                    try {
                        while (!lifeCycle.equals(3)) {
                            try {
                                Thread.sleep(200);
                                buffer = new BufferedReader(new InputStreamReader(new FileInputStream(fifoOutFileName)));
                                break;
                            } catch (Exception ee) {}
                        }

                        while (!lifeCycle.equals(3)) {
                            String line = buffer.readLine();

                            if (line == null) Thread.sleep(200);
                            else {
                                String name = opts.model.substring(7, opts.model.length() -5);

                                if (line.contains("[LOG] Ready")) {
                                    lifeCycle = 1;
                                    callback("_STARTED");
                                }

                                if (line.length() >= name.length() && name.equals(line.substring(1, name.length()+1))) {
                                    callback(line);

                                    if (lifeCycle.equals(21)) {
                                        lifeCycle = 22;
                                        callback("_STOPPED");

                                        if (endApp || closeServiceAfterWakeWordActivation) {
                                            endApp = true;
                                            lifeCycle = 3;
                                        }
                                    }
                                    else if (lifeCycle.equals(1) || lifeCycle.equals(2))
                                        lifeCycle = 21;
                                }
                                else if (line.length() >= 7 && "[ERROR]".equals(line.substring(0, 7)))
                                    callback(line, true);
                                else
                                    Log.d("~= OpenWakeWordService", "stdOut: " + line);
                            }
                        }

                        buffer.close();
                        new File(fifoOutFileName).delete();

                        stopSelf();
                    } catch (Exception e) {
                        Log.e("~= OpenWakeWordService", "stream output error: " + e.toString());
                        try {
                            if (buffer != null) buffer.close();
                        } catch (Exception ee) {}
                    }
                }
            }).start();

            cppStart(extras, Integer.valueOf(extras.getString("delayMS", "0")));

            // by returning this we make sure the service is restarted if the system kills the service
            return Service.START_STICKY;
        }
        else {
            String err = "App is in lifeCycle: " + lifeCycle + ", and is not ready to "
                + (extras.getString("keyword") != null ? "start" : "")
                + (extras.getString("end") != null ? "end" : "")
                + (extras.getString("stop") != null ? "stop" : "");
            callback(err, true);
            return Service.START_REDELIVER_INTENT;
        }
    }

    public void cppStart(Bundle extras) { cppStart(extras, 0); }
    public void cppStart(Bundle extras, int delayMS) {
        if (extras.getString("keyword") != null) {
            opts.model = extras.getString("keyword", "models/alexa_v0.1.onnx");
            opts.threshold = extras.getString("sensitivity", "0.5");
            opts.end_after_activation = true;
            opts.trigger_level = "1";
        }

        Log.d("~= OpenWakeWordService", "openWakeWord cpp and worker STARTING - keyword: " + opts.model + ", sensitivity: " + opts.threshold);

        worker = WorkManager.getInstance(this);
        worker.enqueueUniquePeriodicWork(
            workerName,
            ExistingPeriodicWorkPolicy.KEEP,
            new PeriodicWorkRequest.Builder(OpenWakeWorkWorker.class, 16 /* minimal minutes by documentation */, TimeUnit.MINUTES).build()
        );

        mgr = getResources().getAssets();

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (delayMS > 0) Thread.sleep(delayMS); // If is needed time to mic audio input deallocation

                    openWakeWord(mgr, opts, deviceId, fifoInFileName, fifoOutFileName);

                    worker.cancelUniqueWork(workerName);

                    Log.d("~= OpenWakeWordService", "openWakeWord cpp and worker ENDED");

                    if (stopApp || endApp || lifeCycle.equals(21)) {
                        lifeCycle = 22;
                        callback("_STOPPED");

                        if (endApp || closeServiceAfterWakeWordActivation) {
                            endApp = true;
                            lifeCycle = 3;
                        }
                    }
                    else if (lifeCycle.equals(1) || lifeCycle.equals(2))
                        lifeCycle = 21;
                } catch (Exception e) {
                    callback("c++ error: " + e.toString(), true);
                    cppStart(extras);
                }
            }
        }).start();
    }

    public void callback(String message) { callback(message, false, "openWakeWord"); }
    public void callback(String message, Boolean error) { callback(message, error, "openWakeWord"); }
    public void callback(String message, Boolean error, String requestID) {
        try {
            // Thread.sleep(100);

            if (error == true)
                 Log.e("~= OpenWakeWordService", "callback error: " + message);
            else Log.d("~= OpenWakeWordService", "callback result: " + message);

            Intent intent2 = new Intent(intentFilterBroadcastString);

            intent2.putExtra("requestID", requestID);
            intent2.putExtra(error ? "error" : "result", message);

            // message to app: _STARTED / _RESTARTME / _STOPPED / _ENDED / <wakeWord>
            sendBroadcast(intent2);
        } catch (Exception e) {
            Log.e("~= OpenWakeWordService", "Resolve intent error: " + e.toString());
            throw new RuntimeException(e);
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) { return null; }

    @Override
    public void onDestroy() {
        // Android destroy service automaticly after same time.
        // Android not need call this onDestroy(), that's why you must set worker, which will call this service each 16 minutes.
        if (endApp.equals(false)) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        Thread.sleep(1000);

                        Intent intent2 = new Intent(intentFilterBroadcastString);

                        intent2.putExtra("requestID", "openWakeWord");
                        intent2.putExtra("result", "_RESTARTME");

                        // please restart me
                        sendBroadcast(intent2);
                    } catch (Exception e) {
                        Log.w("~= OpenWakeWordService", "onDestroy() restart error: " + e.toString());
                    }
                }
            }).start();
        } else {
            try { worker.cancelUniqueWork(workerName); } catch (Exception e) {}
            Log.d("~= OpenWakeWordService", "worker END");
        }

        lifeCycle = 4;
        callback("_ENDED");

        stopForeground(true);

        super.onDestroy();

        Log.d("~= OpenWakeWordService", "...DESTROIED");
    }

}
