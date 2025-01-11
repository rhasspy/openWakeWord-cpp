package com.jjassistant;

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.work.Worker;
import androidx.work.WorkerParameters;
import androidx.annotation.NonNull;
import androidx.work.ListenableWorker.Result;

public class OpenWakeWorkWorkerExample extends Worker {
    private final Context context;

    public OpenWakeWorkWorkerExample(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
        this.context = context;
    }

    @NonNull
    @Override
    public Result doWork() {
        Log.d("~= OpenWakeWorkWorker", "sending service restart");

        Intent intent2 = new Intent(OpenWakeWordService.workerName);

        intent2.putExtra("requestID", "openWakeWord");
        intent2.putExtra("result", "_RESTARTME");

        getApplicationContext().sendBroadcast(intent2);

        return Result.success();
    }

    // @Override
    // public void onStopped() {
    //     Log.d("~= OpenWakeWorkWorker", "stopped");
    //     super.onStopped();
    // }
}
