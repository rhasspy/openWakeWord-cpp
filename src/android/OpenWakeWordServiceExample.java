package com.example;

import android.app.Service;
import android.media.AudioManager;
import android.media.AudioDeviceInfo;
import android.content.Context;
import android.content.Intent;
import android.content.ComponentName;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.IBinder;
import android.os.Build;
import android.util.Log;
import android.app.Notification;
import android.content.pm.ServiceInfo;

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

    private static Boolean isRunning = false;
    private static Boolean ending = false;
    private static boolean cppIsRunning = false;

    private static int deviceId = 0;
    private static boolean endAfterActivation = true;
    private static OpenWakeWordOptionsExample opts = new OpenWakeWordOptionsExample();
    private static String fifoOutFileName;
    private static String fifoInFileName;

    public static String workerID = "abc";

    public native void openWakeWord(
        AssetManager mgr,
        OpenWakeWordOptionsExample opts,
        int deviceId,
        String fifoInFileName,
        String fifoOutFileName
    );

    public static native void endOpenWakeWord();

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        Bundle extras = intent.getExtras();
        if (extras == null)
            return Service.START_REDELIVER_INTENT;

        if (extras.getString("end") != null) {
            ending = true;
            stopSelf();
            return Service.START_REDELIVER_INTENT;
        }

        if (isRunning.equals(true)) {
            cppStart(Integer.valueOf(extras.getString("delayMS", "0")));
            return Service.START_REDELIVER_INTENT;
        }

        if (extras.getString("keyword") != null) {
            isRunning = true;

            workerID = extras.getString("workerID");

            String keyword = extras.getString("keyword", "models/alexa_v0.1.onnx");
            String sensitivity = extras.getString("sensitivity", "0.5");

            File dir = getFilesDir();
            if(!dir.exists()) dir.mkdir();
            fifoOutFileName = getFilesDir() + "/fifoOut";
            fifoInFileName = getFilesDir() + "/fifoIn";
        
            Log.d("~= OpenWakeWordService", "onStartCommand - keyword: " + keyword + ", sensitivity: " + sensitivity);

            NotificationCompat.Builder notification = new NotificationCompat.Builder(this, workerID)
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

            WorkManager.getInstance(this).enqueueUniquePeriodicWork(
                workerID,
                ExistingPeriodicWorkPolicy.KEEP,
                new PeriodicWorkRequest.Builder(OpenWakeWorkWorkerExample.class, 16 /* minimal minutes by documentation */, TimeUnit.MINUTES).build()
            );

            new File(fifoOutFileName).delete();

            // stdout reader and callbeck
            new Thread(new Runnable() {
                @Override
                public void run() {
                    BufferedReader buffer = null;
                    try {
                        while (true) {
                            try {
                                Thread.sleep(200);
                                buffer = new BufferedReader(new InputStreamReader(new FileInputStream(fifoOutFileName)));
                                break;
                            } catch (Exception ee) {}
                        }

                        while (true) {
                            String line = buffer.readLine();

                            if (line == null) Thread.sleep(200);
                            else {
                                String name = keyword.substring(7, keyword.length() -5);

                                if (line.length() >= name.length() && name.equals(line.substring(1, name.length()+1)))
                                    callback(line);
                                else if (line.length() >= 7 && "[ERROR]".equals(line.substring(0, 7)))
                                    callback(line, true);
                                else
                                    Log.d("~= OpenWakeWordService", "stdOut: " + line);
                            }
                        }
                    } catch (Exception e) {
                        Log.e("~= OpenWakeWordService", "stream output error: " + e.toString());
                        try {
                            if (buffer != null) buffer.close();
                        } catch (Exception ee) {}
                    }
                }
            }).start();

            AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            AudioDeviceInfo[] devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS);
            for (AudioDeviceInfo device : devices) {
                if (AudioDeviceInfo.TYPE_BUILTIN_MIC == device.getType()) {
                    deviceId = device.getId();
                    break;
                }
            }

            opts.threshold = sensitivity;
            opts.model = keyword;
            opts.end_after_activation = endAfterActivation;
            opts.trigger_level = "1";

            cppStart();
        }

        // by returning this we make sure the service is restarted if the system kills the service
        return Service.START_STICKY;
    }

    public void cppStart() { cppStart(0); }
    public void cppStart(int delayMS) {
        mgr = getResources().getAssets();

        if (cppIsRunning) return;
        else cppIsRunning = true;

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (delayMS > 0) Thread.sleep(delayMS); // If is needed time to mic audio input deallocation
                    openWakeWord(mgr, opts, deviceId, fifoInFileName, fifoOutFileName);
                    Log.d("~= OpenWakeWordService", "openWakeWord END");
                    cppIsRunning = false;
                } catch (Exception e) {
                    Log.e("~= OpenWakeWordService", "c++ error: " + e.toString());
                    callback(e.toString(), true);
                }
            }
        }).start();
    }

    public void callback(String message) { callback(message, false); }
    public void callback(String message, Boolean error) {
        if (error == false)
             Log.d("~= OpenWakeWordService", "result: " + result);
        else Log.e("~= OpenWakeWordService", "error: " + result);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) { return null; }

    @Override
    public void onDestroy() {
        Log.d("~= OpenWakeWordService", "onDestroy");

        endOpenWakeWord();

        isRunning = false;
        cppIsRunning = false;

        stopForeground(true);

        // Android destroy service automaticly after same time.
        // Android not need call this onDestroy(), that's why you must set worker, which will call this service each 16 minutes.
        if (ending == false) {
            ... TODO: there call this service again
        } else {
            try { WorkManager.getInstance(this).cancelUniqueWork(workerID); } catch (Exception e) {}
        }

        super.onDestroy();
    }
}
