"""
Standalone test for Module 2.2: Project Management (Delete Project)
Run from the project root:  python -X utf8 tests/test_module22.py

Tests:
  1. Create two projects.
  2. Delete one project.
  3. Verify the deleted project's workspace folder is removed.
  4. Verify the remaining project is unaffected.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import database, workspace

PROJ_TO_DELETE = "TEST22_DELETE"
PROJ_TO_KEEP = "TEST22_KEEP"

def bootstrap_project(proj: str) -> None:
    """Create a minimal workspace and init DB."""
    proj_path = workspace.get_project_path(proj)
    if os.path.exists(proj_path):
        shutil.rmtree(proj_path)
    workspace.create_project_workspace(proj)
    database.initialize_database(proj)
    database.insert_project(proj, "dummy.pdf", "pdf")

def run() -> None:
    print("\n=== TEST: Delete Project ===")
    
    # 1. Create two projects
    bootstrap_project(PROJ_TO_DELETE)
    bootstrap_project(PROJ_TO_KEEP)
    
    path_delete = workspace.get_project_path(PROJ_TO_DELETE)
    path_keep = workspace.get_project_path(PROJ_TO_KEEP)
    
    assert os.path.exists(path_delete), "Failed to create project to delete"
    assert os.path.exists(path_keep), "Failed to create project to keep"
    print("[OK] Created two projects")
    
    # Verify DB records exist
    assert database.get_project(PROJ_TO_DELETE) is not None
    assert database.get_project(PROJ_TO_KEEP) is not None
    print("[OK] SQLite records created for both projects")
    
    # 2. Delete one project
    workspace.delete_project_workspace(PROJ_TO_DELETE)
    print(f"[OK] Deleted project '{PROJ_TO_DELETE}'")
    
    # 3. Verify deletion
    assert not os.path.exists(path_delete), "Workspace folder still exists"
    assert database.get_project(PROJ_TO_DELETE) is None, "SQLite records still accessible"
    print(f"[OK] Workspace folder and SQLite records removed for '{PROJ_TO_DELETE}'")
    
    # 4. Verify the other project is unaffected
    assert os.path.exists(path_keep), "The kept project workspace was affected"
    assert database.get_project(PROJ_TO_KEEP) is not None, "The kept project DB was affected"
    print(f"[OK] Remaining project '{PROJ_TO_KEEP}' is fully intact")
    
    # Cleanup
    workspace.delete_project_workspace(PROJ_TO_KEEP)
    print("\n" + "=" * 60)
    print("ALL MODULE 2.2 ASSERTIONS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run()
