/**
 * @file FilePickerPanel.hpp
 * @brief Cross-platform modal filesystem browser panel for opening and saving
 *        `.umgebung`, `.scenario.umgebung`, and `.multiverse.umgebung` files.
 *
 * @details
 * Adapted directly from the `FilePickerPanel` present across all branches of
 * `Umgebung-legacy`, upgraded for POSIX/Linux + Windows portability (replacing
 * MSVC-only `strcpy_s` with `std::snprintf`, adding error-code-guarded
 * `std::filesystem::directory_iterator` traversal, alphabetical directory/file
 * sorting, and compound-extension matching for `.scenario.umgebung` and
 * `.multiverse.umgebung`).
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/FilePickerPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/Panel.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace umgebung::ui {

    /**
     * @class FilePickerPanel
     * @brief Modal Dear ImGui popup allowing users to browse directories and
     *        select or name a file to open or save.
     */
    class FilePickerPanel final : public Panel {
    public:
        /** @brief Signature invoked when the user confirms a file selection. */
        using FileSelectedCallback = std::function<void(const std::filesystem::path&)>;

        /** @brief Constructs a closed `FilePickerPanel` rooted at the current working directory. */
        FilePickerPanel();

        /**
         * @brief Opens the file-picker modal with the given title, confirm action,
         *        allowed file extensions/suffixes, and optional starting directory.
         *
         * @param title        Modal dialog window title (e.g., `"Open Scene"`).
         * @param buttonLabel  Primary action button text (e.g., `"Open"` or `"Save"`).
         * @param callback     Callback invoked with the chosen `std::filesystem::path`.
         * @param extensions   Allowed filename suffixes (e.g., `{".umgebung"}`); empty allows all files.
         * @param startPath    Optional directory to navigate into if it exists on disk.
         * @param defaultName  Optional pre-populated filename in the input text box.
         */
        void OpenModal(
            const std::string& title,
            const std::string& buttonLabel,
            FileSelectedCallback callback,
            const std::vector<std::string>& extensions,
            const std::filesystem::path& startPath = {},
            const std::string& defaultName = {}
        );

        /**
         * @brief Renders the modal file-browser popup when active.
         *
         * @param ctx Shared UI context (unused directly, required by `Panel` interface).
         */
        void OnUIRender(UIContext& ctx) override;

    private:
        /** @brief Returns `true` when `filename` matches at least one configured extension suffix. */
        [[nodiscard]] bool MatchesAllowedExtensions(const std::string& filename) const;

        std::string m_title{"File Picker"};
        std::string m_buttonLabel{"Select"};
        FileSelectedCallback m_callback;
        std::vector<std::string> m_extensions;
        std::filesystem::path m_currentPath;
        std::array<char, 256> m_inputBuffer{};
        bool m_shouldOpenPopup{false};
    };

} // namespace umgebung::ui
