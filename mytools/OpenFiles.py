#!/usr/bin/env python
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-15 16:41:06
    @project      : open files with different filename extensions
    @version      : 1.0
    @source file  : open

============================
"""

import sys
import os
            
class OpenFiles():
    """
    open files with different commands
    """
    def __init__(self):
        """
        self.video_types:
            filename extension for video and audio
            types,array
        self.image_types:
            filename extensions for image types,array
        self.text_types:
            filename extensions for text types,array
        self.other_types:
            filename extensions and operation commands,
            dictionary
        """
        super(OpenFiles, self).__init__()
        self.video_types = ["mp4","avi","rmvb","webm","ts",
                            "mp3","ogg","wav","flac"]
        self.image_types = ["jpg","jpeg","gif","png","bmp","icon"]
        self.text_types  = ["txt","py","c","h","html","css","js","gh",
                            "lisp","cpp","go","f","f90",
                            "java","pl","log","tex","bbl","aux",
                            "bib","sh","php"]
        self.other_types = {"md":"typora",
                            "pdf":"evince",
                            "docx":"wps",
                            "ppt":"wpp",
                            "pptx":"wpp",
                            "xls":"et",
                            "xlsx":"et",
                            "doc":"wps"
                            }

    def runCommand(self,program,i):
        """
        open the i-th file withe a specific program
        """
        command = "%s %s"%(program,sys.argv[i])
        os.system(command)
        return

    def run(self,i):
        """
        input: i, index number of the command parameters
        return: None, filename extensions will be classified 
                and open with the corresponding command
        """
        arg = sys.argv[i]
        arg = arg.split(".")
        arg = arg[-1]
        if   arg in self.video_types:
            self.runCommand("mplayer",i)
        elif arg in self.text_types:
            self.runCommand("subl3",i)
        elif arg in self.image_types:
            self.runCommand("eog",i)
        elif arg in self.other_types:
            command = self.other_types[arg]
            self.runCommand(command,i)
        else:
            print("unknown type: " + arg)

        return

    def main(self):
        """
        main function for analyzing
        each command parameters
        """
        if len(sys.argv) == 1:
            print("please input a file")
        else:
            for i in range(1,len(sys.argv)):
                self.run(i)

        return
