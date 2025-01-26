#!/bin/bash
cd /opt/render/project/src
gunicorn wsgi:app 