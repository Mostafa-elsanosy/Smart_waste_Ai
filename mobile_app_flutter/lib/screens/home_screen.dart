import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/scan_result.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentTab = 0;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  int _scannedCount = 0;
  double _co2Saved = 0.0;
  int _categoriesCount = 4;

  @override
  void initState() {
    super.initState();
    _loadStats();
  }

  Future<void> _loadStats() async {
    final stats = await ApiService.getStats();
    if (mounted) {
      setState(() {
        _scannedCount = stats['scanned_items_count'] ?? 0;
        _co2Saved = (stats['co2_saved_kg'] as num?)?.toDouble() ?? 0.0;
        _categoriesCount = stats['categories_count'] ?? 4;
      });
    }
  }

  Future<void> _handleImageSelection(ImageSource source, {String? categoryHint}) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        imageQuality: 85,
      );

      if (pickedFile == null) return;

      setState(() => _isLoading = true);

      final file = File(pickedFile.path);
      final result = await ApiService.scanImage(file, categoryHint: categoryHint);

      setState(() {
        _isLoading = false;
        _scannedCount += 1;
        _co2Saved += 0.25;
      });

      if (!mounted) return;

      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (ctx) => ResultScreen(
            scanResult: result,
            localImageFile: file,
          ),
        ),
      );
    } catch (e) {
      setState(() => _isLoading = false);
      // Fallback demo on error
      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (ctx) => ResultScreen(
            scanResult: ScanResult.samplePaper(),
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: const Color(0xFFF8FAFC),
        appBar: _buildTopBar(),
        body: _isLoading
            ? _buildLoadingOverlay()
            : SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 1. HERO SCAN CARD (Dark Green Gradient)
                    _buildHeroCard(),
                    const SizedBox(height: 16),

                    // 2. KPI METRICS ROW
                    _buildKpiRow(),
                    const SizedBox(height: 20),

                    // 3. SECTION HEADER
                    _buildCategoriesHeader(),
                    const SizedBox(height: 12),

                    // 4. CATEGORIES 2x2 GRID
                    _buildCategoriesGrid(),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
        bottomNavigationBar: _buildBottomNav(),
      ),
    );
  }

  PreferredSizeWidget _buildTopBar() {
    return AppBar(
      backgroundColor: Colors.white,
      elevation: 0,
      scrolledUnderElevation: 0,
      leading: IconButton(
        icon: const Icon(Icons.settings_outlined, color: Color(0xFF1E293B)),
        onPressed: () {},
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.menu_book_outlined, color: Color(0xFF1E293B)),
          onPressed: () {},
        ),
      ],
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: const Color(0xFFD1FAE5),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.recycling_rounded, color: Color(0xFF059669), size: 20),
          ),
          const SizedBox(width: 8),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text(
                'Smart Waste AI',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF0F172A),
                  fontFamily: 'Cairo',
                ),
              ),
              Text(
                'نظام دعم قرار فرز النفايات',
                style: TextStyle(
                  fontSize: 10.5,
                  color: Color(0xFF64748B),
                  fontFamily: 'Cairo',
                ),
              ),
            ],
          ),
        ],
      ),
      centerTitle: true,
    );
  }

  Widget _buildHeroCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF064E3B), Color(0xFF047857)],
          begin: Alignment.topRight,
          end: Alignment.bottomLeft,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF064E3B).withOpacity(0.25),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Text(
              '✨ التعرف الذكي بالرؤية الحاسوبية',
              style: TextStyle(
                fontSize: 11.5,
                fontWeight: FontWeight.w600,
                color: Color(0xFFECFDF5),
                fontFamily: 'Cairo',
              ),
            ),
          ),
          const SizedBox(height: 14),

          // Heading
          const Text(
            'صوّر ما ستلقيه في سلة المهملات',
            style: TextStyle(
              fontSize: 19,
              fontWeight: FontWeight.w800,
              color: Colors.white,
              fontFamily: 'Cairo',
            ),
          ),
          const SizedBox(height: 6),

          // Subtitle
          const Text(
            'يقوم التطبيق بفحص المادة وتصنيفها فوراً إلى Metal أو Paper أو Plastic أو Glass وتحديد السلة المناسبة.',
            style: TextStyle(
              fontSize: 12,
              height: 1.45,
              color: Color(0xFFD1FAE5),
              fontFamily: 'Cairo',
            ),
          ),
          const SizedBox(height: 18),

          // Two Action Buttons
          Row(
            children: [
              // Primary Button (Open Camera)
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () => _handleImageSelection(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt_outlined, size: 18, color: Color(0xFF065F46)),
                  label: const Text(
                    'فتح الكاميرا',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF065F46),
                      fontFamily: 'Cairo',
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    elevation: 0,
                    padding: const EdgeInsets.symmetric(vertical: 11),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),

              // Secondary Button (From Gallery)
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => _handleImageSelection(ImageSource.gallery),
                  icon: const Icon(Icons.image_outlined, size: 18, color: Colors.white),
                  label: const Text(
                    'من المعرض',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      fontFamily: 'Cairo',
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Colors.white, width: 1.2),
                    padding: const EdgeInsets.symmetric(vertical: 11),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildKpiRow() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 12,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildKpiItem(
            value: '$_scannedCount',
            label: 'عنصر تم فحصه',
            icon: Icons.qr_code_scanner_rounded,
            iconColor: const Color(0xFF3B82F6),
          ),
          Container(width: 1, height: 32, color: const Color(0xFFF1F5F9)),
          _buildKpiItem(
            value: '${_co2Saved.toStringAsFixed(1)} كجم',
            label: 'CO2 تم توفيره',
            icon: Icons.eco_outlined,
            iconColor: const Color(0xFF10B981),
          ),
          Container(width: 1, height: 32, color: const Color(0xFFF1F5F9)),
          _buildKpiItem(
            value: '$_categoriesCount مواد',
            label: 'أنواع مصنفة',
            icon: Icons.category_outlined,
            iconColor: const Color(0xFFF59E0B),
          ),
        ],
      ),
    );
  }

  Widget _buildKpiItem({
    required String value,
    required String label,
    required IconData icon,
    required Color iconColor,
  }) {
    return Column(
      children: [
        Icon(icon, size: 20, color: iconColor),
        const SizedBox(height: 6),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            color: Color(0xFF0F172A),
            fontFamily: 'Cairo',
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(
            fontSize: 10.5,
            color: Color(0xFF64748B),
            fontFamily: 'Cairo',
          ),
        ),
      ],
    );
  }

  Widget _buildCategoriesHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text(
          'أقسام وتصنيفات المواد (4 Categories)',
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: Color(0xFF0F172A),
            fontFamily: 'Cairo',
          ),
        ),
        TextButton(
          onPressed: () {},
          child: const Text(
            'عرض الدليل',
            style: TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFF059669),
              fontFamily: 'Cairo',
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCategoriesGrid() {
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 0.95,
      children: [
        // 1. Paper (Blue)
        _buildCategoryCard(
          title: 'ورق وكرتون (Paper)',
          subtitle: 'السلة الزرقاء (مخصصة للورق والكرتون)',
          badgeText: '0 عنصر',
          badgeColor: const Color(0xFF2563EB),
          icon: Icons.menu_book_rounded,
          iconColor: const Color(0xFF2563EB),
          bgColor: const Color(0xFFEFF6FF),
          borderColor: const Color(0xFFDBEAFE),
          onTap: () => _handleImageSelection(ImageSource.gallery, categoryHint: 'Paper'),
        ),

        // 2. Metal (Yellow/Amber)
        _buildCategoryCard(
          title: 'معادن (Metal)',
          subtitle: 'السلة الصفراء / المعدنية',
          badgeText: '0 عنصر',
          badgeColor: const Color(0xFFD97706),
          icon: Icons.inventory_2_outlined,
          iconColor: const Color(0xFFD97706),
          bgColor: const Color(0xFFFFFBEB),
          borderColor: const Color(0xFFFDE68A),
          onTap: () => _handleImageSelection(ImageSource.gallery, categoryHint: 'Metal'),
        ),

        // 3. Plastic (Red/Orange)
        _buildCategoryCard(
          title: 'بلاستيك (Plastic)',
          subtitle: 'السلة البرتقالية / الحمراء (للبلاستيك)',
          badgeText: '0 عنصر',
          badgeColor: const Color(0xFFDC2626),
          icon: Icons.local_drink_outlined,
          iconColor: const Color(0xFFDC2626),
          bgColor: const Color(0xFFFEF2F2),
          borderColor: const Color(0xFFFECACA),
          onTap: () => _handleImageSelection(ImageSource.gallery, categoryHint: 'Plastic'),
        ),

        // 4. Glass (Teal/Green)
        _buildCategoryCard(
          title: 'زجاج (Glass)',
          subtitle: 'السلة الخضراء (مخصصة للزجاجيات)',
          badgeText: '0 عنصر',
          badgeColor: const Color(0xFF059669),
          icon: Icons.wine_bar_outlined,
          iconColor: const Color(0xFF059669),
          bgColor: const Color(0xFFECFDF5),
          borderColor: const Color(0xFFA7F3D0),
          onTap: () => _handleImageSelection(ImageSource.gallery, categoryHint: 'Glass'),
        ),
      ],
    );
  }

  Widget _buildCategoryCard({
    required String title,
    required String subtitle,
    required String badgeText,
    required Color badgeColor,
    required IconData icon,
    required Color iconColor,
    required Color bgColor,
    required Color borderColor,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: borderColor, width: 1.2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: badgeColor,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    badgeText,
                    style: const TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      fontFamily: 'Cairo',
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.all(6),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.04),
                        blurRadius: 4,
                      ),
                    ],
                  ),
                  child: Icon(icon, color: iconColor, size: 20),
                ),
              ],
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFF0F172A),
                    fontFamily: 'Cairo',
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  subtitle,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 10,
                    color: Color(0xFF64748B),
                    fontFamily: 'Cairo',
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Colors.grey.shade200, width: 1)),
      ),
      child: BottomNavigationBar(
        currentIndex: _currentTab,
        onTap: (index) {
          if (index == 1) {
            _handleImageSelection(ImageSource.camera);
          } else {
            setState(() => _currentTab = index);
          }
        },
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        selectedItemColor: const Color(0xFF059669),
        unselectedItemColor: const Color(0xFF94A3B8),
        selectedLabelStyle: const TextStyle(fontFamily: 'Cairo', fontWeight: FontWeight.w700, fontSize: 11),
        unselectedLabelStyle: const TextStyle(fontFamily: 'Cairo', fontSize: 11),
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_filled),
            label: 'الرئيسية',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt_outlined),
            label: 'الكاميرا',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history_rounded),
            label: 'السجل',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.book_outlined),
            label: 'الدليل',
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingOverlay() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: const [
          CircularProgressIndicator(color: Color(0xFF059669)),
          SizedBox(height: 16),
          Text(
            'جاري فحص المادة بالذكاء الاصطناعي...',
            style: TextStyle(fontFamily: 'Cairo', fontWeight: FontWeight.w700, color: Color(0xFF0F172A)),
          ),
        ],
      ),
    );
  }
}
