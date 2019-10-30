#!/usr/bin/env python
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-28 15:01:07
    @project      : Backup for Discuz
    @version      : 1.0
    @source file  : discuz.py

============================
"""
import os

class Discuz():
    """
    make backups from the server
    """
    def __init__(self):
        super(Discuz, self).__init__()
    def run(self):
        """
        backup from the server with three shell commands,
        rsync, ssh and scp
        """
        remote_ip     = "hdp@210.45.125.225"
        dir_disk      = "/home/hdeping/disk/"
        local_dir     = "%sbak_discuzFromServer/"%(dir_disk)
        remote_dir    = "%s:/home/http/"%(remote_ip)
        mysql_command = "mysqldump -u root -pnclxin ultrax > ultrax.sql"
        database      = "%s:ultrax.sql"%(remote_ip)

        commands   = ["rsync -avz %s %s"%(remote_dir,local_dir),
                      "ssh %s '%s'"%(remote_ip,mysql_command),
                      "scp %s %s"%(database,dir_disk)]
        for command in commands:
            print(command)
            os.system(command)

        return

run = Discuz()
run.run()