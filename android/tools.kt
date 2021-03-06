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

// navigation
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController

import android.content.SharedPreferences
import android.content.Context

// network operations
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
// files
import java.io.IOException
import android.Manifest
// Jsoup for crawlers
import org.jsoup.Jsoup


private lateinit  var values_setting:SharedPreferences
private lateinit var  values_editor:SharedPreferences.Editor

private val REQUEST_EXTERNAL_STORAGE = 1
private val PERMISSIONS_STORAGE = arrayOf<String>(
    Manifest.permission.READ_EXTERNAL_STORAGE,
    Manifest.permission.WRITE_EXTERNAL_STORAGE
)
private var urls :List<String> = ArrayList()


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

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    val navView: BottomNavigationView = findViewById(R.id.nav_view)

    val navController = findNavController(R.id.nav_host_fragment)
    // Passing each menu ID as a set of Ids because each
    // menu should be considered as top level destinations.
    val appBarConfiguration = AppBarConfiguration(setOf(
            R.id.navigation_home, 
            R.id.navigation_dashboard, 
            R.id.navigation_notifications))

    setupActionBarWithNavController(navController, appBarConfiguration)
    navView.setupWithNavController(navController)

}
fun browseWeb(qixi:WebView,url:String){
    qixi.settings.setJavaScriptEnabled(true)
    qixi.settings.setUseWideViewPort(true)
    qixi.settings.setLoadWithOverviewMode(true)
    qixi.settings.setUseWideViewPort(true);
    qixi.settings.setLoadWithOverviewMode(true);
    qixi.settings.setSupportZoom(true);
    qixi.settings.setBuiltInZoomControls(true);
    qixi.settings.setDisplayZoomControls(false);
    qixi.settings.setAllowFileAccess(true);
    qixi.settings.setLoadsImagesAutomatically(true);
    qixi.settings.setDefaultTextEncodingName("utf-8")
    qixi.setLayerType(View.LAYER_TYPE_HARDWARE,null)

    qixi.loadUrl(url)
    qixi.setWebViewClient(object : WebViewClient() {
        override fun shouldOverrideUrlLoading(view: WebView, url: String): Boolean {
            //使用WebView加载显示url
            view.loadUrl(url)
            //返回true
            return true
        }
        override fun onPageFinished(view: WebView, url: String) {
            //使用WebView加载显示url
            var count = values_setting.getInt(type,10)
            var script = """
                javascript:
                count = ${count};
                change_content();
            """.trimIndent()
            //返回true
            view.loadUrl(script)
            }

    })
}

fun applyChange(qixi:WebView,type:String){
    val script = "javascript:get_count()"
    qixi.evaluateJavascript(script,object : ValueCallback<String>{
        override fun onReceiveValue(count: String) {
            values_editor.putInt(type,count.toInt())
            values_editor.apply()
        }
    })
}


fun getPreferences(){
    val prefer = "settings"
    values_setting = this.getSharedPreferences(prefer, Context.MODE_PRIVATE )
    values_editor  = values_setting.edit()
    values_setting.getString("str1","str1")
    values_setting.getInt("int1",0)

}


fun alert(message:String){
    val alertdialogbuilder: AlertDialog.Builder = Builder(this)
    alertdialogbuilder.setMessage(message)
    alertdialogbuilder.setPositiveButton("确定", null)
    alertdialogbuilder.setNeutralButton("取消", null)
    val alertdialog1: AlertDialog = alertdialogbuilder.create()
    alertdialog1.show()

}

fun checkNetwork(){
    Thread(Runnable{
        try {
            var s: Socket? = null
            if (s == null) {
                s = Socket()
            }
            var ip = "114.114.114.114"
            val host: InetAddress = InetAddress.getByName(ip) 
            s.connect(InetSocketAddress(host, 53), 5000) //goo gle:53
            s.close()
        } catch (e: IOException) {
            alert("无法联网,请检查网络连接")
        }
    })

}

fun runJs(web:WebView,script:String){

    web.evaluateJavascript(script,object : ValueCallback<String>{
        override fun onReceiveValue(count: String) {
            Log.d("result",count)
        }
    })
}

fun writeTxtFile(content: String, filePath: String, fileName: String, append: Boolean): Boolean {
    var flag: Boolean = true
    val thisFile = File("$filePath/$fileName")
    try {
        if (!thisFile.parentFile.exists()) {
            thisFile.parentFile.mkdirs()
        }
        val fw = FileWriter("$filePath/$fileName", append)
        fw.write(content)
        fw.close()
    } catch (e: IOException) {
        e.printStackTrace()
    }
    return flag
}

fun verifyStoragePermissions(activity: Activity?) {
    // Check if we have write permission
    val permission =
        ActivityCompat.checkSelfPermission(
            activity!!,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    if (permission != PackageManager.PERMISSION_GRANTED) {
        // We don't have permission so prompt the user
        ActivityCompat.requestPermissions(
            activity,
            PERMISSIONS_STORAGE,
            REQUEST_EXTERNAL_STORAGE
        )
    }
}

fun loadImageFromUrl(url:String) Drawable{
    var img:InputStream = URL(url).getContent() as InputStream
    var d:Drawable = Drawable.createFromStream(img,"image")
    return d
}

fun update(){
    Thread(Runnable {
        try{
            var text = Jsoup.connect(url).timeout(2000).get()
            var image_url = text.select("#bigpicimg").src
            var links = text.select(".pages>ul>li")
            url = links[links.size - 1]
            runOnUiThread(Runnable {
                
            })
        }

        catch(e:Exception){
            runOnUiThread(Runnable {
                alert("无法打开网页！")
            })
        }
    }).start()
}

import android.database.sqlite.SQLiteDatabase  
import android.database.sqlite.SQLiteOpenHelper  
import android.content.ContentValues  
import android.database.Cursor  
import android.database.sqlite.SQLiteException  
  
//creating the database logic, extending the SQLiteOpenHelper base class  
class DatabaseHandler(context: Context): SQLiteOpenHelper(context,DATABASE_NAME,null,DATABASE_VERSION) {  
    companion object {  
        private val DATABASE_VERSION = 1  
        private val DATABASE_NAME = "EmployeeDatabase"  
        private val TABLE_CONTACTS = "EmployeeTable"  
        private val KEY_ID = "id"  
        private val KEY_NAME = "name"  
        private val KEY_EMAIL = "email"  
    }  
    override fun onCreate(db: SQLiteDatabase?) {  
       // TODO("not implemented") //To change body of created functions use File | Settings | File Templates.  
       //creating table with fields  
        val CREATE_CONTACTS_TABLE = ("CREATE TABLE if not exists" + TABLE_CONTACTS + "("  
                + KEY_ID + " INTEGER PRIMARY KEY," + KEY_NAME + " TEXT,"  
                + KEY_EMAIL + " TEXT" + ")")  
        db?.execSQL(CREATE_CONTACTS_TABLE)  
    }  
  
    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {  
      //  TODO("not implemented") //To change body of created functions use File | Settings | File Templates.  
        db!!.execSQL("DROP TABLE IF EXISTS " + TABLE_CONTACTS)  
        onCreate(db)  
    }  
  
  
    //method to insert data  
    fun addEmployee(emp: EmpModelClass):Long{  
        val db = this.writableDatabase  
        val contentValues = ContentValues()  
        contentValues.put(KEY_ID, emp.userId)  
        contentValues.put(KEY_NAME, emp.userName) // EmpModelClass Name  
        contentValues.put(KEY_EMAIL,emp.userEmail ) // EmpModelClass Phone  
        // Inserting Row  
        val success = db.insert(TABLE_CONTACTS, null, contentValues)  
        //2nd argument is String containing nullColumnHack  
        db.close() // Closing database connection  
        return success  
    }  
    //method to read data  
    fun viewEmployee():List<EmpModelClass>{  
        val empList:ArrayList<EmpModelClass> = ArrayList<EmpModelClass>()  
        val selectQuery = "SELECT  * FROM $TABLE_CONTACTS"  
        val db = this.readableDatabase  
        var cursor: Cursor? = null  
        try{  
            cursor = db.rawQuery(selectQuery, null)  
        }catch (e: SQLiteException) {  
            db.execSQL(selectQuery)  
            return ArrayList()  
        }  
        var userId: Int  
        var userName: String  
        var userEmail: String  
        if (cursor.moveToFirst()) {  
            do {  
                userId = cursor.getInt(cursor.getColumnIndex("id"))  
                userName = cursor.getString(cursor.getColumnIndex("name"))  
                userEmail = cursor.getString(cursor.getColumnIndex("email"))  
                val emp= EmpModelClass(userId = userId, userName = userName, userEmail = userEmail)  
                empList.add(emp)  
            } while (cursor.moveToNext())  
        }  
        return empList  
    }  
    //method to update data  
    fun updateEmployee(emp: EmpModelClass):Int{  
        val db = this.writableDatabase  
        val contentValues = ContentValues()  
        contentValues.put(KEY_ID, emp.userId)  
        contentValues.put(KEY_NAME, emp.userName) // EmpModelClass Name  
        contentValues.put(KEY_EMAIL,emp.userEmail ) // EmpModelClass Email  
  
        // Updating Row  
        val success = db.update(TABLE_CONTACTS, contentValues,"id="+emp.userId,null)  
        //2nd argument is String containing nullColumnHack  
        db.close() // Closing database connection  
        return success  
    }  
    //method to delete data  
    fun deleteEmployee(emp: EmpModelClass):Int{  
        val db = this.writableDatabase  
        val contentValues = ContentValues()  
        contentValues.put(KEY_ID, emp.userId) // EmpModelClass UserId  
        // Deleting Row  
       val success = db.delete(TABLE_CONTACTS,"id="+emp.userId,null)  
        //2nd argument is String containing nullColumnHack  
        db.close() // Closing database connection  
        return success  
    }  


import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.Socket
fun connectSocket(){
    go.setOnClickListener{
        Thread(Runnable {
            try{
                val socket = Socket("114.214.200.176", 1991)
                val outputStream: OutputStream = socket.getOutputStream()
                // create a data output stream from the output stream so we can send data through it
                val dataOutputStream = DataOutputStream(outputStream)
                // write the message we want to send
                var value = input.text.toString()
                dataOutputStream.writeUTF(value)
                dataOutputStream.flush() // send the message
                socket.shutdownOutput()
                // get data from the server
                val inputStream: InputStream = socket.getInputStream()
                val dataInputStream = DataInputStream(inputStream)
                var message:String = ""
                var res:Byte = 0
                try {
                    while (true) {
                        res = dataInputStream.readByte()
                        message += res.toChar()
                    }
                } catch (e: java.lang.Exception) {
                    println("something wrong with reading data")
                    System.out.println(value)
                }
                socket.close()
                runOnUiThread(Runnable {
                    output.setText(message)
                })
            }
            catch(e:Exception){
                uiAlert("无法连接远程服务器！")
            }
        }).start()
    }
}
