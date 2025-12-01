from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainingSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('training_name', models.CharField(max_length=255, verbose_name='ชื่อการ train')),
                ('model_name', models.CharField(max_length=100, verbose_name='ชื่อโมเดล')),
                ('dataset_count', models.IntegerField(default=1, verbose_name='จำนวน dataset')),
                ('epochs', models.IntegerField(default=10, verbose_name='จำนวน epochs')),
                ('status', models.CharField(choices=[('pending', 'รอการ train'), ('training', 'กำลัง train'), ('completed', 'train เสร็จแล้ว'), ('failed', 'train ล้มเหลว')], default='pending', max_length=20, verbose_name='สถานะการ train')),
                ('accuracy', models.FloatField(blank=True, null=True, verbose_name='ความแม่นยำ')),
                ('loss', models.FloatField(blank=True, null=True, verbose_name='Loss')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='วันที่สร้าง')),
                ('started_at', models.DateTimeField(blank=True, null=True, verbose_name='เวลา เริ่มต้น')),
                ('completed_at', models.DateTimeField(blank=True, null=True, verbose_name='เวลา เสร็จสิ้น')),
                ('error_message', models.TextField(blank=True, null=True, verbose_name='ข้อความ error')),
                ('model_path', models.CharField(blank=True, max_length=500, null=True, verbose_name='เส้นทางโมเดล')),
            ],
            options={
                'verbose_name': 'Training Session',
                'verbose_name_plural': 'Training Sessions',
                'ordering': ['-created_at'],
            },
        ),
    ]
