<?php
/**
 * Recursively deletes all files and subfolders inside the given folder.
 *
 * @param string $folder Path of the directory to be deleted.
 * 
 * @return void
 */
function deleteFolderContents($folder) {
    if (!is_dir($folder)) return;
    $files = scandir($folder);
    foreach ($files as $file) {
        if ($file === '.' || $file === '..') continue;
        $path = $folder . DIRECTORY_SEPARATOR . $file;
        if (is_dir($path)) {
            deleteFolderContents($path);
            rmdir($path);
        } else {
            unlink($path);
        }
    }
}
//Folder is not deleted, it should be deleted afterwards using rmdir($folder);

/**
 * Recursively copies all files and subdirectories from a source directory to a destination directory.
 *
 * @param string $origen Path of the source directory to copy from.
 * @param string $destination Path of the destination directory to copy to.
 * 
 * @return void
 */
function copyRecursive($source, $destination) {
    if (!is_dir($source)) return;
    @mkdir($destination);
    $files = scandir($source);
    foreach ($files as $file) {
        if ($file === '.' || $file === '..') continue;
        $source_path = $source . DIRECTORY_SEPARATOR . $file;
        $destination_path = $destination . DIRECTORY_SEPARATOR . $file;
        if (is_dir($source_path)) {
            copyRecursive($source_path, $destination_path);
        } else {
            copy($source_path, $destination_path);
        }
    }
}
?>