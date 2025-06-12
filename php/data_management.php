<?php
/**
 * Loads the data contained in a JSON file.
 *
 * @param string $file Path of the file to be read. Ex: assets/db/data.json
 * 
 * @return array|false Returns an associative array on success, or false on failure.
 */
function loadJSONFile(string $file): array|false
{
    if (!file_exists($file)) {
        error_log("Error: JSON file not found at " . $file);
        return false;
    }

    $json = file_get_contents($file);

    if ($json === false) {
        error_log("Error: Could not read contents of file " . $file);
        return false;
    }

    $data = json_decode($json, true);

    if (json_last_error() !== JSON_ERROR_NONE) {
        error_log("Error decoding JSON from file " . $file . ": " . json_last_error_msg());
        return false;
    }

    return $data;
}
?>

Using foreach we can loop the data and show it in HTML:
<ul>
    <?php foreach ($loadJSONFile("data.json") as $person): ?>
    <li><?php echo $person["name"]; ?></li>
    <?php endforeach; ?>
</ul>

we can modify the data and save it:
<?php
$data[0]['name'] = "Albert";
$new_json = json_encode($data, JSON_PRETTY_PRINT);
file_put_contents('new_data.json', $new_json);
?>