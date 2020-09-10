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
fun toast(text:String){
    Toast.makeText(this,text,Toast.LENGTH_LONG).show()
}
fun readAssetsFile(name:String): String{
    val input = assets.open(name)
    val size = input.available()
    val buffer = ByteArray(size)
    input.read(buffer)
    return String(buffer)
}

fun str2Float(str:String):Float{
    try {
        var num = str.toFloat()
        return num
    }
    catch (e:Exception){
        output.setText("请输入数字")
        return 0.0f
    }
}

fun judgeTriangle(a:Float,b:Float,c:Float): Boolean{
    if (a > 0 && b> 0 && c>0 && a+b>c && a+c>b && c+b>a){
        output.setTextColor(R.color.black)
        output.setText("$a $b $c 可以构成三角形")
        return true
    }
    else{
        output.setTextColor(R.color.red)
        output.setText("$a $b $c 无法构成三角形")
        side1.setText("")
        side2.setText("")
        side3.setText("")
        angle1.setText("")
        angle2.setText("")
        angle3.setText("")
        outer_radius.setText("")
        inner_radius.setText("")
        area_value.setText("")
        return false
    }
}

fun getTriangleArea(a:Float,b:Float,c:Float){
    var p = (a+b+c)/2
    if (judgeTriangle(a,b,c)){
        var s = sqrt(p*(p-a)*(p-b)*(p-c))
        area_value.setText(String.format("%.2f",s))
        var r1 = a*b*c/s/4
        outer_radius.setText(String.format("%.2f",r1))
        var r2 = s/p
        inner_radius.setText(String.format("%.2f",r2))
        var A = a/2/r1
        A = asin(A)*180.0f/PI.toFloat()
        angle1.setText(String.format("%.2f",A))
        A = b/2/r1
        A = asin(A)*180.0f/PI.toFloat()
        angle2.setText(String.format("%.2f",A))
        A = c/2/r1
        A = asin(A)*180.0f/PI.toFloat()
        angle3.setText(String.format("%.2f",A))
    }
}