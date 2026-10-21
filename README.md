# ♻️ Smart Waste AI & Decision Support System (DSS)
### نظام إدارة وفرز النفايات الذكي المدعوم بالذكاء الاصطناعي ونظام دعم اتخاذ القرار

---

## 📖 نظرة عامة على المشروع (Project Overview)

منصة متكاملة وموحدة لإدارة وفرز النفايات الذكية تجمع بين:
1. **محرك الذكاء الاصطناعي (YOLOv8 Computer Vision)**: الكشف والتصنيف الدقيق لأربعة تصنيفات رئيسية:
   - 🥫 **معادن (Metal)**: علب الكانز، رقائق الألومنيوم، أدوات ومخلفات معدنية.
   - 🧴 **بلاستيك (Plastic)**: زجاجات PET، حاويات، أكياس وبوليمرات.
   - 🍶 **زجاج (Glass)**: برطمانات زجاجية، زجاجات، عبوات عطور وزجاجيات.
   - 📦 **ورق وكرتون (Paper)**: صناديق كرتون، أوراق، مجلات ومستندات.
2. **محرك دعم القرار (Decision Support System - DSS)**:
   - **قواعد الفرز التلقائي**: توجيه كل مادة للحاوية المخصصة لها بواسطة مشغلات ميكانيكية دقيقة (Pneumatic ejectors, Robotic delta grippers, Air knives).
   - **قاعدة حماية النقاء (Purity Safeguard)**: توجيه أي عنصر تقل ثقة الذكاء الاصطناعي فيه عن 60% إلى محطة الفحص اليدوي لمنع تلوث دفعات التدوير.
   - **قاعدة حماية الامتلاء والفيضان (Overflow Protection)**: فحص مستوى امتلاء الحاوية لحظياً وإعادة التوجيه إلى الصوامع الاحتياطية في حال وصول الامتلاء إلى ≥ 95%.
   - **قاعدة سلامة المناولة (Handling Safety)**: تخفيض سرعة خط السير للزجاج القابل للكسر (≤ 0.8 m/s) واستخدام مخمدات هيدروليكية للمعادن الثقيلة.
3. **تتبع الحاويات وإنترنت الأشياء (IoT & Telemetry Tracking)**:
   - مراقبة الحجم (L)، الوزن (kg)، نسبة الامتلاء بالموجات فوق الصوتية (%)، درجة الحرارة، وحالة البطارية.
   - شجرة حالات التنبيه: `NORMAL` (<75%)، `WARNING` (75–89%)، `CRITICAL` (≥90%)، `OVERFLOW` (≥100%).
4. **تحسين مسارات شاحنات التجميع (TSP Logistics Optimizer)**:
   - حل مسألة بائع متجول ذكية لحساب أقصر مسار يزور الحاويات الحرجة فقط، مع حساب كمية الوقود الموفرة (L) وخفض انبعاثات ثاني أكسيد الكربون (CO₂ kg).
5. **واجهات مستخدم موحدة واحترافية**:
   - **لوحة تحكم الويب العصرية (Modern Glassmorphic Web Dashboard)**: لوحة تحكم تفاعلية متجاوبة داكنة تعمل مباشرة عبر المتصفح.
   - **تطبيق الجوال الذكي (Mobile App - Flutter)**: واجهة عربية RTL متطابقة مع التصميم ومحاكي ويب مباشر يعمل في المتصفح (`/mobile`).

---

## 📁 هيكل مجلدات المشروع بعد التنظيف وإلغاء الازدواجية (Clean Structure)

```
dss2/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # إعدادات النظام وعتبات الحاويات والشاحنات
│       ├── main.py                     # مسارات FastAPI وتأمين نقاط النهاية
│       ├── schemas.py                  # نماذج Pydantic للتحقق من صحة البيانات
│       ├── dss/                        # نواة نظام دعم اتخاذ القرار (DSS Core)
│       │   ├── __init__.py
│       │   ├── sorting_engine.py       # محرك وقواعد الفرز التلقائي وسجل التدقيق
│       │   ├── fill_tracker.py         # تتبع حساسات الامتلاء والتنبيهات
│       │   └── route_optimizer.py      # خوارزمية TSP وتحليلات توفير الكربون
│       └── services/
│           ├── __init__.py
│           └── waste_service.py        # واجهة الربط الموحدة بين YOLO و DSS
├── models/
│   ├── __init__.py
│   ├── yolo_detector.py                # محرك الكشف متعدد النماذج وتحليل السمات اللونية
│   ├── waste_yolov8_best.pt            # نموذج YOLOv8 مخصص لفرز 18 صنف نفايات
│   ├── yolov8n.pt                      # أوزان YOLOv8n الأساسية للأجسام المتنوعة
│   └── yolov8n_waste.pt                # نموذج مدرب على 12 صنفاً من المواد القابلة للتدوير
├── frontend/
│   └── web/                            # لوحة تحكم Glassmorphism ومحاكي الجوال الموحد
│       ├── index.html                  # واجهة الويب الرئيسية
│       ├── mobile.html                 # محاكي تطبيق الجوال بالعربية RTL
│       ├── css/style.css               # تنسيقات Glassmorphism الداكنة
│       └── js/app.js                   # المنطق التفاعلي وعرض المسارات
├── data/
│   └── sample_images/                  # عينات صور الاختبار المعتمدة (Metal, Plastic, Glass, Paper)
├── mobile_app_flutter/                 # تطبيق الجوال المعتمد الكامل (Flutter / Dart)
│   ├── pubspec.yaml
│   └── lib/ (main.dart, screens/, widgets/, models/, services/)
├── tests/
│   └── test_api_dss.py                 # 10 اختبارات شاملة للنظام والنماذج
├── .env.example                        # نموذج المتغيرات البيئية
├── .gitignore                          # استبعاد ملفات الكاش والبيئات الافتراضية
├── pytest.ini                          # إعدادات المسار التلقائي للاختبارات
├── requirements.txt                    # متطلبات وحزم بايثون المعتمدة
└── run.py                              # مشغل موحد للمشروع بكافة مكوناته
```

---

## 🛠️ متطلبات التشغيل (System Prerequisites)

- **Python**: الإصدار `3.10` أو `3.11` أو `3.12` (مُختبر ومعتمد بنجاح).
- **نظام التشغيل**: Windows 10/11 أو Linux أو macOS.
- **Flutter SDK** (اختياري فقط في حال رغبت بتجميع وبناء تطبيق الجوال محلياً).

---

## 🚀 خطوات التثبيت والتشغيل بالتفصيل (Step-by-Step Installation)

### الخطوة 1: الدخول لمجلد المشروع (Open Directory)
```bash
cd dss2
```

### الخطوة 2: إنشاء وتفعيل البيئة الافتراضية (Virtual Environment)
- **على Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **على Windows (CMD)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
- **على Linux أو macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### الخطوة 3: تثبيت الحزم المطلوبة (Install Dependencies)
```bash
pip install -r requirements.txt
```

### الخطوة 4: التحقق من التثبيت وتشغيل الاختبارات الآلية (Run Tests)
```bash
pytest tests/test_api_dss.py -v
```
*(ستظهر النتيجة: `10 passed in ~17s` بنسبة نجاح 100%)*

---

## 💻 تشغيل التطبيق (Running the Application)

يتم تشغيل المشروع بالكامل بأمر واحد بسيط:
```bash
python run.py
```
أو لفتح المتصفح تلقائياً عند التشغيل:
```bash
python run.py --open-browser
```

### روابط الوصول بعد التشغيل:
- 🌐 **لوحة تحكم الويب الرئيسية**: [http://localhost:8000](http://localhost:8000)
- 📱 **محاكي تطبيق الجوال الذكي (عربي RTL)**: [http://localhost:8000/mobile](http://localhost:8000/mobile)
- 📚 **توثيق الواجهات التفاعلي (Swagger Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📱 تشغيل تطبيق الجوال (Flutter Mobile App)

1. **المحاكي المباشر في المتصفح**: لا يتطلب أي برامج إضافية، افتح [http://localhost:8000/mobile](http://localhost:8000/mobile).
2. **بناء التطبيق عبر Flutter**:
   ```bash
   cd mobile_app_flutter
   flutter pub get
   flutter run
   ```

---

## 🔒 إجراءات الأمان والحماية المطبقة (Security & Hardening)

1. **الحماية من التلاعب بالمسارات (Path Traversal Protection)**: تأمين نقطة النهاية `/api/sample-images/{filename}`.
2. **الحد الأقصى لحجم الملفات (Max Payload Limit)**: تحديد سقف حجم الصورة المرفوعة بـ `15 ميجابايت`.
3. **التخزين المؤقت الذكي فائق السرعة (In-Memory LRU Cache)**: زمن الاستجابة للصور المتكررة أقل من **3 مللي ثانية**.
4. **خوارزمية منع تكرار الصناديق (IoU NMS)**: دمج الصناديق المتراكبة من النماذج العصبية المتعددة.
5. **معالجة غير متزامنة خفيفة (Asynchronous Threadpool Offloading)**: تفريغ أحمال النماذج إلى خيوط منفصلة لضمان استجابة السيرفر الفورية.

---

## 📡 جدول واجهات برمجة التطبيقات (API Reference)

| الطريقة | المسار (Endpoint) | الوصف |
|---|---|---|
| `GET` | `/` | عرض لوحة تحكم الويب الرئيسية Glassmorphic Web App |
| `GET` | `/mobile` | عرض محاكي تطبيق الجوال بالعربية RTL |
| `GET` | `/docs` | واجهة Swagger التفاعلية لاختبار الـ API |
| `GET` | `/api/health` | فحص حالة السيرفر، المحرك العصبي ونظام DSS |
| `POST` | `/api/waste/classify` | رفع صورة لتصنيفها بـ YOLOv8 وتطبيق قواعد فرز DSS |
| `POST` | `/api/v1/scan` | نقطة نهاية فحص الجوال وإرجاع التوصيات العربية والنسب |
| `GET` | `/api/v1/categories` | جلب أصناف النفايات الأربعة مع ألوان السلات والرموز |
| `GET` | `/api/v1/mobile/stats` | جلب مؤشرات الأداء الحيوية (عناصر مفحوصة، CO₂ موفر) |
| `POST` | `/api/waste/manual-entry` | محاكاة فرز عنصر يدوياً واختبار قواعد DSS دون صورة |
| `GET` | `/api/containers` | استرجاع بيانات وحساسات جميع حاويات المدينة الذكية |
| `PUT` | `/api/containers/{bin_id}` | تحديث حجم حاوية أو إعادة تصفيرها بعد التفريغ |
| `POST` | `/api/containers/empty-all` | تصفير جميع الحاويات في الشبكة دفعة واحدة |
| `GET` | `/api/dss/route-optimization` | حساب مسار شاحنة التجميع الأمثل TSP وحساب توفير الكربون |
| `POST` | `/api/simulation/tick` | محاكاة تراكم النفايات لمدة ساعة واحدة في المدينة |
| `GET` | `/api/statistics` | الإحصائيات الإجمالية، دقة الفرز، وكميات الكربون المعاد تدويرها |
| `GET` | `/api/sample-images` | قائمة عينات الصور الجاهزة للاختبار |
