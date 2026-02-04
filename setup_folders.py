"""
フォルダ構造セットアップスクリプト
プログラム実行前に必要なフォルダ構造を作成します
"""

from pathlib import Path
import sys

def create_folder_structure():
    """必要なフォルダ構造を作成"""
    
    # 作成するフォルダのリスト
    folders = [
        "models",              # YOLOモデルファイル
        "videos/input",        # 入力動画
        "videos/output",       # 出力動画
        "results",             # 統計データ（CSV、JSON）
        "docs"                 # ドキュメント
    ]
    
    print("=" * 60)
    print("📁 MICHI-AI - フォルダ構造セットアップ")
    print("=" * 60)
    print()
    
    created_folders = []
    existing_folders = []
    
    for folder in folders:
        folder_path = Path(folder)
        
        if folder_path.exists():
            existing_folders.append(folder)
            print(f"✓ 既に存在: {folder}")
        else:
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(folder)
                print(f"✓ 作成しました: {folder}")
            except Exception as e:
                print(f"✗ エラー: {folder} の作成に失敗しました - {e}")
                return False
    
    print()
    print("=" * 60)
    print("📊 セットアップ結果")
    print("=" * 60)
    print(f"新規作成: {len(created_folders)} 個")
    print(f"既存: {len(existing_folders)} 個")
    print(f"合計: {len(folders)} 個")
    print()
    
    if created_folders:
        print("🎉 フォルダ構造のセットアップが完了しました！")
    else:
        print("✓ すべてのフォルダは既に存在しています")
    
    print()
    print("📝 次のステップ:")
    print("  1. videos/input/ に処理したい動画ファイルを配置")
    print("  2. models/ にYOLOモデルファイル（yolov8n.pt等）を配置")
    print("     ※初回実行時は自動ダウンロードされます")
    print("  3. main_gui.py または main_cui.py を実行")
    print()
    
    return True

def main():
    """メイン処理"""
    try:
        success = create_folder_structure()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ 処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 予期しないエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
