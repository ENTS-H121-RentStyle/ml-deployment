# How to Run?
Pastikan anda sudah install Cron dan Postfix SMTP, lalu jalankan perintah ini
```
crontab -e
```

Atur jadwal dijalankannya script seperti ini, yaitu jalan setiap hari jam 8 malam
```
0 20 * * * /path/to/script/ml.sh
```
