import 'package:flutter/material.dart';

class MaterialProbability {
  final String material;
  final String materialAr;
  final double percentage;
  final Color color;
  final bool isPrimary;

  MaterialProbability({
    required this.material,
    required this.materialAr,
    required this.percentage,
    required this.color,
    required this.isPrimary,
  });

  factory MaterialProbability.fromJson(Map<String, dynamic> json) {
    Color parseColor(String hexColor) {
      hexColor = hexColor.replaceAll('#', '');
      if (hexColor.length == 6) {
        hexColor = 'FF$hexColor';
      }
      return Color(int.parse('0x$hexColor'));
    }

    return MaterialProbability(
      material: json['material'] ?? '',
      materialAr: json['material_ar'] ?? '',
      percentage: (json['percentage'] as num?)?.toDouble() ?? 0.0,
      color: parseColor(json['color'] ?? '#3B82F6'),
      isPrimary: json['is_primary'] ?? false,
    );
  }
}

class DSSRecommendation {
  final String binLabelAr;
  final String targetBinAr;
  final String binColorName;
  final Color binColor;
  final String icon;
  final String adviceAr;
  final String decisionStatus;

  DSSRecommendation({
    required this.binLabelAr,
    required this.targetBinAr,
    required this.binColorName,
    required this.binColor,
    required this.icon,
    required this.adviceAr,
    required this.decisionStatus,
  });

  factory DSSRecommendation.fromJson(Map<String, dynamic> json) {
    Color parseColor(String hexColor) {
      hexColor = hexColor.replaceAll('#', '');
      if (hexColor.length == 6) hexColor = 'FF$hexColor';
      return Color(int.parse('0x$hexColor'));
    }

    return DSSRecommendation(
      binLabelAr: json['bin_label_ar'] ?? 'حاوية المهملات المناسبة:',
      targetBinAr: json['target_bin_ar'] ?? 'السلة الزرقاء (مخصصة للورق والكرتون)',
      binColorName: json['bin_color_name'] ?? 'Blue',
      binColor: parseColor(json['bin_color_hex'] ?? '#2563EB'),
      icon: json['icon'] ?? 'trash-blue',
      adviceAr: json['advice_ar'] ?? '',
      decisionStatus: json['decision_status'] ?? 'APPROVED',
    );
  }
}

class ScanResult {
  final String detectedCategory;
  final String categoryBadge;
  final String titleAr;
  final String descriptionAr;
  final bool recyclable;
  final String statusBadge;
  final Color statusBadgeColor;
  final double overallConfidence;
  final List<MaterialProbability> probabilityBreakdown;
  final DSSRecommendation dssRecommendation;
  final String? annotatedImageBase64;
  final double inferenceTimeMs;

  ScanResult({
    required this.detectedCategory,
    required this.categoryBadge,
    required this.titleAr,
    required this.descriptionAr,
    required this.recyclable,
    required this.statusBadge,
    required this.statusBadgeColor,
    required this.overallConfidence,
    required this.probabilityBreakdown,
    required this.dssRecommendation,
    this.annotatedImageBase64,
    required this.inferenceTimeMs,
  });

  factory ScanResult.fromJson(Map<String, dynamic> json) {
    Color parseColor(String hexColor) {
      hexColor = hexColor.replaceAll('#', '');
      if (hexColor.length == 6) hexColor = 'FF$hexColor';
      return Color(int.parse('0x$hexColor'));
    }

    var list = json['probability_breakdown'] as List? ?? [];
    List<MaterialProbability> probs = list.map((i) => MaterialProbability.fromJson(i)).toList();

    return ScanResult(
      detectedCategory: json['detected_category'] ?? 'Paper',
      categoryBadge: json['category_badge'] ?? 'ورق وكرتون (Paper)',
      titleAr: json['title_ar'] ?? 'منتج ورقي / كرتون (Paper / Cardboard)',
      descriptionAr: json['description_ar'] ?? '',
      recyclable: json['recyclable'] ?? true,
      statusBadge: json['status_badge'] ?? 'قابل للتدوير',
      statusBadgeColor: parseColor(json['status_badge_color'] ?? '#10B981'),
      overallConfidence: (json['overall_confidence'] as num?)?.toDouble() ?? 72.0,
      probabilityBreakdown: probs,
      dssRecommendation: DSSRecommendation.fromJson(json['dss_recommendation'] ?? {}),
      annotatedImageBase64: json['annotated_image_base64'],
      inferenceTimeMs: (json['inference_time_ms'] as num?)?.toDouble() ?? 18.5,
    );
  }

  // Factory providing fallback default matching the design screenshot
  factory ScanResult.samplePaper() {
    return ScanResult(
      detectedCategory: 'Paper',
      categoryBadge: 'ورق وكرتون (Paper)',
      titleAr: 'منتج ورقي / كرتون (Paper / Cardboard)',
      descriptionAr: 'اكتشاف سطح غير لامع (Matte) مع تدرجات لونية دافئة ونسيج ألياف يطابق الورق والكرتون.',
      recyclable: true,
      statusBadge: 'قابل للتدوير',
      statusBadgeColor: const Color(0xFF10B981),
      overallConfidence: 72.0,
      probabilityBreakdown: [
        MaterialProbability(material: 'Metal', materialAr: 'معادن', percentage: 23.9, color: const Color(0xFFF59E0B), isPrimary: false),
        MaterialProbability(material: 'Paper', materialAr: 'ورق', percentage: 40.5, color: const Color(0xFF3B82F6), isPrimary: true),
        MaterialProbability(material: 'Glass', materialAr: 'زجاج', percentage: 12.8, color: const Color(0xFF10B981), isPrimary: false),
        MaterialProbability(material: 'Plastic', materialAr: 'بلاستيك', percentage: 22.8, color: const Color(0xFFEF4444), isPrimary: false),
      ],
      dssRecommendation: DSSRecommendation(
        binLabelAr: 'حاوية المهملات المناسبة:',
        targetBinAr: 'السلة الزرقاء (مخصصة للورق والكرتون)',
        binColorName: 'Blue',
        binColor: const Color(0xFF2563EB),
        icon: 'trash-blue',
        adviceAr: 'يرجى التأكد من جفاف الورق والكرتون.',
        decisionStatus: 'APPROVED',
      ),
      inferenceTimeMs: 18.5,
    );
  }
}
