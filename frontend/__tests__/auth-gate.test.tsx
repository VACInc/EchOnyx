import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthGate } from "@/components/auth-gate";
import { api } from "@/lib/api";
import { renderWithProviders } from "./test-utils";

vi.mock("@/lib/api", () => ({
  api: {
    getAuthSession: vi.fn(),
    getOidcLoginUrl: vi.fn(() => "http://localhost:8000/api/auth/oidc/login"),
    login: vi.fn(),
    logout: vi.fn(),
    setupAuth: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

function session(overrides: Partial<Awaited<ReturnType<typeof api.getAuthSession>>> = {}) {
  return {
    authenticated: false,
    setup_required: false,
    actor_label: null,
    password_enabled: true,
    oidc: {
      enabled: false,
      provider_name: null,
      login_path: null,
    },
    ...overrides,
  };
}

describe("AuthGate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.getAuthSession.mockResolvedValue(session());
    mockedApi.login.mockResolvedValue(session({ authenticated: true }));
    mockedApi.setupAuth.mockResolvedValue(session({ authenticated: true }));
  });

  it("renders setup mode when setup is required", async () => {
    mockedApi.getAuthSession.mockResolvedValueOnce(session({ setup_required: true }));

    renderWithProviders(
      <AuthGate>
        <div>Protected app</div>
      </AuthGate>,
    );

    expect(await screen.findByRole("heading", { name: "Secure EchOnyx" })).toBeInTheDocument();
    expect(screen.getByLabelText("Confirm password")).toBeInTheDocument();
    expect(screen.queryByText("Protected app")).not.toBeInTheDocument();
  });

  it("renders login mode otherwise", async () => {
    renderWithProviders(
      <AuthGate>
        <div>Protected app</div>
      </AuthGate>,
    );

    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.queryByText("Protected app")).not.toBeInTheDocument();
  });

  it("shows the OIDC button when enabled", async () => {
    mockedApi.getAuthSession.mockResolvedValueOnce(
      session({
        password_enabled: false,
        oidc: {
          enabled: true,
          provider_name: "Acme SSO",
          login_path: "/api/auth/oidc/login",
        },
      }),
    );

    renderWithProviders(
      <AuthGate>
        <div>Protected app</div>
      </AuthGate>,
    );

    expect(await screen.findByRole("button", { name: "Sign in with Acme SSO" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("submits the password and displays login errors", async () => {
    const user = userEvent.setup();
    mockedApi.login.mockRejectedValueOnce(new Error("Wrong password"));

    renderWithProviders(
      <AuthGate>
        <div>Protected app</div>
      </AuthGate>,
    );

    await user.type(await screen.findByLabelText("Password"), "correct horse");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(mockedApi.login.mock.calls[0]?.[0]).toBe("correct horse");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Wrong password"));
  });
});
