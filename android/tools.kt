/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-10 10:22:41
    @project      : tools for android develpment
    @version      : 1.0
    @source file  : tools.kt

============================

*/

import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AlertDialog.Builder
import android.util.Log
import android.webkit.*
import android.os.Bundle
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_main.*


fun changeBgImage(yuanyuan:ImageView){
    var images:ArrayList<Int> = ArrayList()
    images.add(R.drawable.yuanyuan6465)
    images.add(R.drawable.yuanyuan6475)
    images.add(R.drawable.yuanyuan6596)
    images.add(R.drawable.yuanyuan6597)
    images.add(R.drawable.yuanyuan6601)
    images.add(R.drawable.yuanyuan6602)
    var count = 0
    yuanyuan.setOnClickListener{
        count += 1
        count = count% images.size
        yuanyuan.setImageResource(images[count])
    }
}