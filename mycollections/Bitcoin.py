#!/usr/bin/env python
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-12-11 15:49:24
    @project      : wrapper for bitcoin-cli
    @version      : 1.0
    @source file  : Bitcoin.py

============================
"""

import json
import os


class Bitcoin():
    """
    wrapper for bitcoin-cli"""
    def __init__(self):
        """
        self.command:
            command for bitcoin-cli
        """
        super(Bitcoin, self).__init__()
        self.command = "bitcoin-cli getblock `bitcoin-cli getblockhash %d`"
        return
    def run(self):
        for i in range(600000)
        return

        