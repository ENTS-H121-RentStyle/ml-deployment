# How to Run?
Pastikan anda sudah install Cron dan Postfix SMTP, lalu jalankan perintah ini
```
crontab -e
```

Atur jadwal dijalankannya script seperti ini
```
0 0 * * 3 /path/to/script/ml.sh
```
