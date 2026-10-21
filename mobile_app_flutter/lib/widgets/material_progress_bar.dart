import 'package:flutter/material.dart';
import '../models/scan_result.dart';

class MaterialProgressBarRow extends StatelessWidget {
  final MaterialProbability item;

  const MaterialProgressBarRow({
    Key? key,
    required this.item,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.5),
      child: Row(
        children: [
          // Percentage (Left side)
          SizedBox(
            width: 44,
            child: Text(
              '${item.percentage.toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 12,
                fontWeight: item.isPrimary ? FontWeight.w700 : FontWeight.w500,
                color: item.isPrimary ? const Color(0xFF2563EB) : const Color(0xFF374151),
                fontFamily: 'Cairo',
              ),
            ),
          ),
          const SizedBox(width: 8),

          // Horizontal Progress Bar
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: Container(
                height: 7.0,
                color: const Color(0xFFF3F4F6),
                child: FractionallySizedBox(
                  alignment: Alignment.centerLeft,
                  widthFactor: (item.percentage / 100.0).clamp(0.0, 1.0),
                  child: Container(
                    decoration: BoxDecoration(
                      color: item.color,
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),

          // Material Name (Right side)
          SizedBox(
            width: 50,
            child: Text(
              item.material,
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: 12.5,
                fontWeight: item.isPrimary ? FontWeight.w800 : FontWeight.w600,
                color: item.isPrimary ? const Color(0xFF2563EB) : const Color(0xFF374151),
                fontFamily: 'Cairo',
              ),
            ),
          ),
        ],
      ),
    );
  }
}
