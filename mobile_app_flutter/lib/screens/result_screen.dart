import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import '../models/scan_result.dart';
import '../widgets/circular_confidence_gauge.dart';
import '../widgets/material_progress_bar.dart';

class ResultScreen extends StatelessWidget {
  final ScanResult scanResult;
  final File? localImageFile;

  const ResultScreen({
    Key? key,
    required this.scanResult,
    this.localImageFile,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: const Color(0xFFF8FAFC),
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 0,
          centerTitle: true,
          leading: IconButton(
            icon: const Icon(Icons.arrow_forward_ios, size: 20, color: Color(0xFF1E293B)),
            onPressed: () => Navigator.of(context).pop(),
          ),
          title: const Text(
            'نتيجة الفحص الذكي',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Color(0xFF0F172A),
              fontFamily: 'Cairo',
            ),
          ),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Column(
            children: [
              // 1. TOP CARD (Detected Object & Preview)
              _buildTopCard(context),
              const SizedBox(height: 14),

              // 2. MIDDLE CARD (AI Material Analysis & Confidence)
              _buildMiddleCard(context),
              const SizedBox(height: 14),

              // 3. BOTTOM CARD (DSS Recommendation Output)
              _buildBottomCard(context),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  // TOP CARD
  Widget _buildTopCard(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Image with Floating Badge
          Stack(
            children: [
              ClipRRect(
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(24),
                  topRight: Radius.circular(24),
                ),
                child: SizedBox(
                  width: double.infinity,
                  height: 210,
                  child: _buildImageWidget(),
                ),
              ),

              // Top Right Floating Status Badge: "قابل للتدوير"
              Positioned(
                top: 14,
                right: 14,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xFF10B981),
                    borderRadius: BorderRadius.circular(999),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.12),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.recycling, size: 16, color: Colors.white),
                      const SizedBox(width: 4),
                      Text(
                        scanResult.statusBadge,
                        style: const TextStyle(
                          fontSize: 12.5,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                          fontFamily: 'Cairo',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),

          // Details Section
          Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Category Tag
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEFF6FF),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    scanResult.categoryBadge,
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF2563EB),
                      fontFamily: 'Cairo',
                    ),
                  ),
                ),
                const SizedBox(height: 10),

                // Title & Book Icon Row
                Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Expanded(
                      child: Text(
                        scanResult.titleAr,
                        style: const TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF0F172A),
                          fontFamily: 'Cairo',
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Container(
                      width: 46,
                      height: 46,
                      decoration: BoxDecoration(
                        color: const Color(0xFFEFF6FF),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(Icons.menu_book_rounded, color: Color(0xFF2563EB), size: 24),
                    ),
                  ],
                ),
                const SizedBox(height: 8),

                // Description
                Text(
                  scanResult.descriptionAr,
                  style: const TextStyle(
                    fontSize: 12.5,
                    height: 1.5,
                    color: Color(0xFF64748B),
                    fontFamily: 'Cairo',
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // MIDDLE CARD (AI Material Analysis & Confidence)
  Widget _buildMiddleCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(5),
                decoration: BoxDecoration(
                  color: const Color(0xFFECFDF5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.bar_chart_rounded, size: 20, color: Color(0xFF10B981)),
              ),
              const SizedBox(width: 8),
              const Text(
                'تحليل نسب المواد وتأكيد الذكاء الاصطناعي',
                style: TextStyle(
                  fontSize: 14.5,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF0F172A),
                  fontFamily: 'Cairo',
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Content Row: Horizontal Bars on Left, Circular Gauge on Right
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Material Bars
              Expanded(
                flex: 6,
                child: Column(
                  children: scanResult.probabilityBreakdown.map((item) {
                    return MaterialProgressBarRow(item: item);
                  }).toList(),
                ),
              ),
              const SizedBox(width: 14),

              // Circular Progress Ring (72% دقة التصنيف)
              Expanded(
                flex: 4,
                child: Center(
                  child: CircularConfidenceGauge(
                    percentage: scanResult.overallConfidence,
                    size: 98,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // BOTTOM CARD (DSS Recommendation Output)
  Widget _buildBottomCard(BuildContext context) {
    final dss = scanResult.dssRecommendation;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(5),
                decoration: BoxDecoration(
                  color: const Color(0xFFEFF6FF),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.fact_check_outlined, size: 20, color: Color(0xFF2563EB)),
              ),
              const SizedBox(width: 8),
              const Text(
                'قرار الفرز الموصى به (DSS Recommendation)',
                style: TextStyle(
                  fontSize: 14.5,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF0F172A),
                  fontFamily: 'Cairo',
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),

          // Callout Box
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: const Color(0xFFEFF6FF),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFDBEAFE)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        dss.binLabelAr,
                        style: const TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w500,
                          color: Color(0xFF64748B),
                          fontFamily: 'Cairo',
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        dss.targetBinAr,
                        style: const TextStyle(
                          fontSize: 14.5,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF2563EB),
                          fontFamily: 'Cairo',
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                const Icon(
                  Icons.delete_outline_rounded,
                  size: 28,
                  color: Color(0xFF2563EB),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageWidget() {
    if (scanResult.annotatedImageBase64 != null && scanResult.annotatedImageBase64!.isNotEmpty) {
      try {
        final cleanB64 = scanResult.annotatedImageBase64!.contains(',')
            ? scanResult.annotatedImageBase64!.split(',')[1]
            : scanResult.annotatedImageBase64!;
        return Image.memory(base64Decode(cleanB64), fit: BoxFit.cover);
      } catch (_) {}
    }
    if (localImageFile != null && localImageFile!.existsSync()) {
      return Image.file(localImageFile!, fit: BoxFit.cover);
    }
    // Default high-resolution paper image
    return Image.network(
      'https://images.unsplash.com/photo-1586075010923-2dd4570fb338?auto=format&fit=crop&w=800&q=80',
      fit: BoxFit.cover,
      errorBuilder: (ctx, err, stack) => Container(
        color: const Color(0xFFE2E8F0),
        child: const Icon(Icons.image, size: 50, color: Colors.grey),
      ),
    );
  }
}
